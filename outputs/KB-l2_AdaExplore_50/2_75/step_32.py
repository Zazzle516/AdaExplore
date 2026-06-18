import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_min_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N,
    eps,
    NUM_GROUPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0)
    w = tl.load(W_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)

    x_2d = tl.reshape(x, (NUM_GROUPS, GROUP_SIZE))
    mean = tl.sum(x_2d, axis=1) / GROUP_SIZE
    diff = x_2d - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + eps)

    x_norm = diff * rstd[:, None]
    x_norm_flat = tl.reshape(x_norm, (BLOCK_N,))
    y = x_norm_flat * w + b

    y_masked = tl.where(mask, y, float('inf'))
    min_val = tl.min(y_masked, axis=0)

    tl.store(Out_ptr + row, min_val)


def triton_gn_min(x, weight, bias, num_groups, eps):
    M, N = x.shape
    GROUP_SIZE = N // num_groups
    out = torch.empty((M,), device=x.device, dtype=x.dtype)
    grid = (M,)
    gn_min_kernel[grid](
        x, weight, bias, out,
        M, N, eps,
        NUM_GROUPS=num_groups,
        GROUP_SIZE=GROUP_SIZE,
        BLOCK_N=N,
        num_warps=8,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups

        gemm = nn.Linear(in_features, out_features)
        self.gemm_weight = nn.Parameter(gemm.weight.detach().clone())
        self.gemm_bias = nn.Parameter(gemm.bias.detach().clone())

        gn = nn.GroupNorm(num_groups, out_features)
        self.gn_weight = nn.Parameter(gn.weight.detach().clone())
        self.gn_bias = nn.Parameter(gn.bias.detach().clone())
        self.eps = 1e-5

        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        # Use cuBLAS for the GEMM — it's hard to beat on 4090 at these sizes.
        y = torch.addmm(self.gemm_bias, x, self.gemm_weight.t())
        m = triton_gn_min(y, self.gn_weight, self.gn_bias, self.num_groups, self.eps)
        m = m.view(1, 1, -1, 1)
        out = m + self.bias
        return out