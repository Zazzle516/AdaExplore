import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_swish_bias_gn_kernel(
    Y_ptr, Bias_ptr, Gamma_ptr, Beta_ptr,
    M, N: tl.constexpr, C: tl.constexpr, G: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # One program per (row, group). Computes swish(y) + bias for that
    # group's channels, then groupnorm-normalizes within the group.
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    col_base = pid_g * C
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    cols = col_base + offs_c

    y_ptrs = Y_ptr + pid_m * N + cols
    y = tl.load(y_ptrs, mask=mask_c, other=0.0)
    bb = tl.load(Bias_ptr + cols, mask=mask_c, other=0.0)

    # Swish + bias
    y = y * tl.sigmoid(y) + bb

    # Compute mean / var over the group
    y_f = y.to(tl.float32)
    # zero out masked lanes for reduction
    y_f_masked = tl.where(mask_c, y_f, 0.0)
    s = tl.sum(y_f_masked, axis=0)
    ss = tl.sum(y_f_masked * y_f_masked, axis=0)
    inv_c = 1.0 / C
    mean = s * inv_c
    var = ss * inv_c - mean * mean
    invstd = 1.0 / tl.sqrt(var + EPS)

    gamma = tl.load(Gamma_ptr + cols, mask=mask_c, other=0.0)
    beta = tl.load(Beta_ptr + cols, mask=mask_c, other=0.0)

    out = (y_f - mean) * invstd * gamma + beta
    tl.store(y_ptrs, out, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features
        G = self.num_groups
        C = N // G

        # GEMM via cuBLAS (addmm: bias + x @ W^T)
        Y = torch.addmm(self.matmul.bias, x, self.matmul.weight.t())

        BLOCK_C = triton.next_power_of_2(C)
        if BLOCK_C < 16:
            BLOCK_C = 16

        grid = (M, G)
        fused_swish_bias_gn_kernel[grid](
            Y, self.bias, self.group_norm.weight, self.group_norm.bias,
            M, N, C, G,
            self.eps,
            BLOCK_C=BLOCK_C,
            num_warps=2,
            num_stages=2,
        )
        return Y