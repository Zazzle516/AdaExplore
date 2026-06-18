import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_reduce_splitk_kernel(
    x_ptr, w_ptr, b_ptr, sub_ptr, row_sum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SPLIT_N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_split = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # split N range
    n_per_split = N // SPLIT_N
    n_begin = pid_split * n_per_split
    n_end = n_begin + n_per_split

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(n_begin, n_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = mask_m[:, None] & mask_k[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
            w_mask = mask_k[:, None] & mask_n[None, :]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile)

        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        sub_vals = tl.load(sub_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias_vals[None, :] - sub_vals[None, :]
        acc = tl.where(mask_n[None, :], acc, 0.0)

        row_sum += tl.sum(acc, axis=1)

    # atomic add to global row_sum
    tl.atomic_add(row_sum_ptr + offs_m, row_sum, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=16),
    ],
    key=['K'],
)
@triton.jit
def gelu_add_kernel(
    x_ptr, row_sum_ptr, out_ptr,
    M, K, inv_N,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    s_val = tl.load(row_sum_ptr + pid_m) * inv_N
    inv_sqrt2 = 0.7071067811865475
    gelu_s = 0.5 * s_val * (1.0 + tl.erf(s_val * inv_sqrt2))

    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
    mask_k = offs_k < K
    x_vals = tl.load(x_ptr + pid_m * K + offs_k, mask=mask_k, other=0.0)
    out_vals = x_vals + gelu_s
    tl.store(out_ptr + pid_m * K + offs_k, out_vals, mask=mask_k)


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

        W = self.gemm.weight.contiguous()
        bias = self.gemm.bias.contiguous() if self.gemm.bias is not None else torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        # split-K (split-N really) factor — choose based on N
        if N >= 4096:
            SPLIT_N = 8
        elif N >= 1024:
            SPLIT_N = 4
        else:
            SPLIT_N = 1

        # ensure N divisible
        while SPLIT_N > 1 and (N % SPLIT_N != 0):
            SPLIT_N //= 2

        row_sum = torch.zeros((B,), device=x.device, dtype=torch.float32)

        grid1 = lambda meta: (triton.cdiv(B, meta['BLOCK_M']), SPLIT_N)
        gemm_reduce_splitk_kernel[grid1](
            x, W, bias, sub, row_sum,
            B, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SPLIT_N=SPLIT_N,
        )

        out = torch.empty_like(x)
        inv_N = 1.0 / float(N)
        grid2 = lambda meta: (B, triton.cdiv(K, meta['BLOCK']))
        gelu_add_kernel[grid2](x, row_sum, out, B, K, inv_N)

        return out