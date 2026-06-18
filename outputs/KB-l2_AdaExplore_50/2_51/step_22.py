import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Non-split GEMM-reduce: each program handles BLOCK_M rows, loops over full N.
# Computes row_sum[m] = sum_n (W @ x + bias - sub)[m, n].
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sub_reduce_kernel(
    x_ptr, w_ptr, b_ptr, sub_ptr, row_sum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

            w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            acc += tl.dot(x_tile, w_tile)

        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        sub_vals = tl.load(sub_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias_vals[None, :] - sub_vals[None, :]
        acc = tl.where(mask_n[None, :], acc, 0.0)

        row_sum += tl.sum(acc, axis=1)

    tl.store(row_sum_ptr + offs_m, row_sum, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=16),
    ],
    key=['K'],
)
@triton.jit
def add_gelu_mean_kernel(
    x_ptr, row_sum_ptr, out_ptr,
    M, K, inv_N,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # Load row_sum, compute mean -> GELU
    s = tl.load(row_sum_ptr + pid_m)
    mean_val = s * inv_N
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * mean_val * (1.0 + tl.erf(mean_val * inv_sqrt2))

    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
    mask_k = offs_k < K
    x_vals = tl.load(x_ptr + pid_m * K + offs_k, mask=mask_k, other=0.0)
    tl.store(out_ptr + pid_m * K + offs_k, x_vals + g, mask=mask_k)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        K = x.shape[1]
        N = self.out_features

        W = self.gemm.weight.contiguous()  # (N, K)
        if self.gemm.bias is not None:
            bias = self.gemm.bias.contiguous()
        else:
            bias = torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        row_sum = torch.empty((B,), device=x.device, dtype=torch.float32)

        grid1 = lambda meta: (triton.cdiv(B, meta['BLOCK_M']),)
        gemm_sub_reduce_kernel[grid1](
            x, W, bias, sub, row_sum,
            B, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        out = torch.empty_like(x)
        inv_N = 1.0 / float(N)
        grid2 = lambda meta: (B, triton.cdiv(K, meta['BLOCK']))
        add_gelu_mean_kernel[grid2](x, row_sum, out, B, K, inv_N)

        return out