import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Split-K style: each program handles a (BLOCK_M rows) x (BLOCK_N cols) tile.
# It computes the GEMM tile, adds bias - sub, sums over N tile, and atomically
# adds to a per-row partial-sum accumulator. This exposes B * (N/BLOCK_N) parallelism.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_subtract_rowreduce_splitk_kernel(
    x_ptr, w_ptr, b_ptr, sub_ptr, rowsum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    x_base = x_ptr + offs_m[:, None] * stride_xm
    w_base = w_ptr + offs_n[None, :] * stride_wn

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K

        x_ptrs = x_base + k_idx[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = w_base + k_idx[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        acc += tl.dot(x_tile, w_tile)

    bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    sub_vals = tl.load(sub_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias_vals[None, :] - sub_vals[None, :]
    acc = tl.where(mask_n[None, :], acc, 0.0)

    partial = tl.sum(acc, axis=1)

    # Atomic add into per-row accumulator
    tl.atomic_add(rowsum_ptr + offs_m, partial, mask=mask_m)


@triton.jit
def finalize_gelu_kernel(
    rowsum_ptr, s_ptr, M, inv_N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    val = tl.load(rowsum_ptr + offs, mask=mask, other=0.0)
    mean_val = val * inv_N
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * mean_val * (1.0 + tl.erf(mean_val * inv_sqrt2))
    tl.store(s_ptr + offs, gelu_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def add_scalar_kernel(
    x_ptr, s_ptr, out_ptr,
    M, K,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    s_val = tl.load(s_ptr + pid_m)
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
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

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        K = x.shape[1]
        N = self.out_features

        W = self.gemm.weight.contiguous()
        if self.gemm.bias is not None:
            bias = self.gemm.bias.contiguous()
        else:
            bias = torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        # Per-row partial-sum accumulator (zeroed; we atomic-add into it)
        rowsum = torch.zeros((B,), device=x.device, dtype=torch.float32)

        grid1 = lambda meta: (triton.cdiv(B, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        gemm_subtract_rowreduce_splitk_kernel[grid1](
            x, W, bias, sub, rowsum,
            B, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        s = torch.empty((B,), device=x.device, dtype=torch.float32)
        BLOCK_FIN = 256
        grid_fin = (triton.cdiv(B, BLOCK_FIN),)
        finalize_gelu_kernel[grid_fin](rowsum, s, B, 1.0 / N, BLOCK=BLOCK_FIN)

        out = torch.empty_like(x)
        grid2 = lambda meta: (B, triton.cdiv(K, meta['BLOCK']))
        add_scalar_kernel[grid2](x, s, out, B, K)

        return out