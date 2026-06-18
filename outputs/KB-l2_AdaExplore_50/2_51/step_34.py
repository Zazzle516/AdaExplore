import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def gemm_reduce_split_kernel(
    x_ptr, w_ptr, b_ptr, sub_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_split = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    N_per_split = (N + SPLIT_K - 1) // SPLIT_K
    n_begin = pid_split * N_per_split
    n_end = tl.minimum(n_begin + N_per_split, N)

    for n_start in range(n_begin, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)

            w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)

            acc += tl.dot(x_tile, w_tile)

        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        sub_vals = tl.load(sub_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias_vals[None, :] - sub_vals[None, :]

        acc = tl.where(mask_n[None, :], acc, 0.0)
        row_sum += tl.sum(acc, axis=1)

    out_ptrs = partial_ptr + pid_split * M + offs_m
    tl.store(out_ptrs, row_sum, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 2048}, num_warps=8),
        triton.Config({'BLOCK_K': 4096}, num_warps=8),
        triton.Config({'BLOCK_K': 8192}, num_warps=16),
        triton.Config({'BLOCK_K': 4096}, num_warps=16),
        triton.Config({'BLOCK_K': 2048}, num_warps=16),
    ],
    key=['K'],
)
@triton.jit
def finalize_and_add_kernel(
    partial_ptr, x_ptr, out_ptr,
    M, K, N_total,
    SPLIT_K_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    sum_val = 0.0
    for s in tl.static_range(0, SPLIT_K_C):
        v = tl.load(partial_ptr + s * M + pid_m)
        sum_val += v

    mean_val = sum_val / N_total
    inv_sqrt2 = 0.7071067811865475
    s_val = 0.5 * mean_val * (1.0 + tl.erf(mean_val * inv_sqrt2))

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    x_vals = tl.load(x_ptr + pid_m * K + offs_k, mask=mask_k, other=0.0)
    out_vals = x_vals + s_val
    tl.store(out_ptr + pid_m * K + offs_k, out_vals, mask=mask_k)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self.SPLIT_K = 16

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        K = x.shape[1]
        N = self.out_features

        W = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous() if self.gemm.bias is not None else torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        SPLIT_K = self.SPLIT_K
        partial = torch.empty((SPLIT_K, B), device=x.device, dtype=torch.float32)

        grid1 = lambda meta: (triton.cdiv(B, meta['BLOCK_M']), SPLIT_K)
        gemm_reduce_split_kernel[grid1](
            x, W, bias, sub, partial,
            B, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SPLIT_K=SPLIT_K,
        )

        out = torch.empty_like(x)
        grid2 = lambda meta: (B, triton.cdiv(K, meta['BLOCK_K']))
        finalize_and_add_kernel[grid2](
            partial, x, out,
            B, K, N,
            SPLIT_K_C=SPLIT_K,
        )

        return out