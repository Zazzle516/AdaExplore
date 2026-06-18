import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_max_sumexp_kernel(
    x_ptr, w_ptr, b_ptr,
    max_ptr, sum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # This kernel computes y = x @ W^T + b for a tile of rows, and
    # contributes partial max and sumexp via atomics. But atomics for max/sum
    # in stable softmax style are tricky. Instead, we do per-row reduction
    # within the kernel by iterating over all N tiles.
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # First pass: compute max over the row by iterating all N tiles
    row_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)

    # Pass 1: find max
    for n_idx in range(0, num_n_tiles):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
        n_mask = offs_n[None, :] < N
        acc = tl.where(n_mask, acc, -float('inf'))
        tile_max = tl.max(acc, axis=1)
        row_max = tl.maximum(row_max, tile_max)

    # Pass 2: compute sum of exp(x - max)
    for n_idx in range(0, num_n_tiles):
        offs_n = n_idx * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
            w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, tl.trans(w))
        b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + b[None, :]
        n_mask = offs_n[None, :] < N
        e = tl.exp(acc - row_max[:, None])
        e = tl.where(n_mask, e, 0.0)
        tile_sum = tl.sum(e, axis=1)
        row_sum = row_sum + tile_sum

    m_mask = offs_m < M
    tl.store(max_ptr + offs_m, row_max, mask=m_mask)
    tl.store(sum_ptr + offs_m, row_sum, mask=m_mask)


@triton.jit
def finalize_kernel(
    max_ptr, sum_ptr, out_ptr,
    M,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    rm = tl.load(max_ptr + offs, mask=mask, other=0.0)
    rs = tl.load(sum_ptr + offs, mask=mask, other=1.0)
    x = rm + tl.log(rs)
    # LeakyReLU twice (slope 0.01)
    x = tl.where(x >= 0, x, x * 0.01)
    x = tl.where(x >= 0, x, x * 0.01)
    # GELU twice (exact via erf)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(out_ptr + offs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous().cuda()
        if self.linear.bias is not None:
            b = self.linear.bias.contiguous().cuda()
        else:
            b = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)

        M, K = x.shape
        N = W.shape[0]

        row_max = torch.empty(M, device=x.device, dtype=torch.float32)
        row_sum = torch.empty(M, device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        gemm_max_sumexp_kernel[grid](
            x, W, b,
            row_max, row_sum,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        out = torch.empty(M, 1, device=x.device, dtype=x.dtype)
        BLOCK = 256
        grid2 = (triton.cdiv(M, BLOCK),)
        finalize_kernel[grid2](row_max, row_sum, out, M, BLOCK=BLOCK)
        return out