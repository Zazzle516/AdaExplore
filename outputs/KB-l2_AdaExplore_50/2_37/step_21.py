import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gemm_swish_bias_gn_kernel(
    X_ptr, W_ptr, B_ptr, Bias_ptr, Gamma_ptr, Beta_ptr, Y_ptr,
    M, N, K,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    # Program: one (row-tile, group) pair
    # BLOCK_N is fixed to GROUP_SIZE so each program owns exactly one group
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_g * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    # GEMM
    acc = tl.zeros((BLOCK_M, GROUP_SIZE), dtype=tl.float32)
    x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
    w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]

    for k in range(0, K, BLOCK_K):
        k_mask = (offs_k[None, :] + k) < K
        x = tl.load(x_ptrs + k, mask=mask_m[:, None] & k_mask, other=0.0)
        w_mask = (offs_k[:, None] + k) < K
        w = tl.load(w_ptrs + k, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

    # Add bias from Linear (B) and the extra Parameter bias (Bias)
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias_extra = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    # Swish: x * sigmoid(x)
    acc = acc * tl.sigmoid(acc)

    # Add the extra bias param (post-swish, per the reference)
    acc = acc + bias_extra[None, :]

    # GroupNorm: compute mean/var over this group (GROUP_SIZE elements per row)
    mean = tl.sum(acc, axis=1) / GROUP_SIZE
    diff = acc - mean[:, None]
    var = tl.sum(diff * diff, axis=1) / GROUP_SIZE
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Affine
    gamma = tl.load(Gamma_ptr + offs_n, mask=offs_n < N, other=1.0)
    beta = tl.load(Beta_ptr + offs_n, mask=offs_n < N, other=0.0)

    out = (acc - mean[:, None]) * rstd[:, None] * gamma[None, :] + beta[None, :]

    # Store
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, out, mask=mask_m[:, None] & (offs_n[None, :] < N))


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.group_size = out_features // num_groups

        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.matmul.weight.contiguous()
        B = self.matmul.bias.contiguous()
        bias_extra = self.bias.contiguous()
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_K = 32
        GROUP_SIZE = self.group_size

        grid = (triton.cdiv(M, BLOCK_M), self.num_groups)

        fused_gemm_swish_bias_gn_kernel[grid](
            x, W, B, bias_extra, gamma, beta, y,
            M, N, K,
            GROUP_SIZE=GROUP_SIZE,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
            EPS=self.eps,
            num_warps=4,
            num_stages=3,
        )
        return y