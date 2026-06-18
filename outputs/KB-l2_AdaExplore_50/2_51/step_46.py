import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Fused GEMM + subtract + row-sum reduction.
# Each program handles a tile of BLOCK_M rows, loops over N in BLOCK_N chunks
# and over K in BLOCK_K chunks, accumulating sum_n(X@W^T + b - sub) into a
# [BLOCK_M] register vector. Avoids materializing the [M,N] intermediate.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sub_rowreduce_kernel(
    X_ptr, Wt_ptr, b_ptr, sub_ptr, RowSum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_k_tiles = tl.cdiv(K, BLOCK_K)

    for n_tile in range(0, num_n_tiles):
        n_idx = n_tile * BLOCK_N + offs_n
        n_mask = n_idx < N

        # accumulator for this N-tile [BLOCK_M, BLOCK_N]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = Wt_ptr + (offs_k[:, None] * stride_wtk + n_idx[None, :] * stride_wtn)

        for k_tile in range(0, num_k_tiles):
            k_remaining = K - k_tile * BLOCK_K
            if k_remaining >= BLOCK_K:
                x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
                w = tl.load(w_ptrs, mask=n_mask[None, :], other=0.0)
            else:
                k_mask = offs_k < k_remaining
                x = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
                w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wtk

        b = tl.load(b_ptr + n_idx, mask=n_mask, other=0.0)
        s = tl.load(sub_ptr + n_idx, mask=n_mask, other=0.0)
        acc = acc + (b - s)[None, :]
        acc = tl.where(n_mask[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    tl.store(RowSum_ptr + offs_m, row_acc, mask=m_mask)


# Fused mean / GELU / residual-add: scalar = gelu(rowsum / N), then out = scalar + original_x.
@triton.jit
def fused_mean_gelu_residual_kernel(
    X_ptr, RowSum_ptr, OUT_ptr,
    M, K, N,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K

    rs = tl.load(RowSum_ptr + pid_m)
    s = rs / N
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * s * (1.0 + tl.math.erf(s * inv_sqrt2))

    x = tl.load(X_ptr + pid_m * K + offs_k, mask=mask, other=0.0)
    out = x + g
    tl.store(OUT_ptr + pid_m * K + offs_k, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._Wt_cache = None
        self._Wt_version = -1
        self._Wt_device = None

    def _get_Wt(self):
        W = self.gemm.weight
        if (self._Wt_cache is None
                or self._Wt_version != W._version
                or self._Wt_device != W.device):
            self._Wt_cache = W.t().contiguous()
            self._Wt_version = W._version
            self._Wt_device = W.device
        return self._Wt_cache

    def forward(self, x):
        x = x.contiguous()
        original_x = x  # we don't actually need a clone; we only read from x
        M, K = x.shape
        N = self.out_features

        Wt = self._get_Wt()  # [K, N] contiguous
        if self.gemm.bias is not None:
            b = self.gemm.bias.contiguous()
        else:
            b = torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        RowSum = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        gemm_sub_rowreduce_kernel[grid](
            x, Wt, b, sub, RowSum,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
        )

        out = torch.empty_like(x)
        BLOCK_K = 1024
        grid2 = (M, triton.cdiv(K, BLOCK_K))
        fused_mean_gelu_residual_kernel[grid2](
            original_x, RowSum, out, M, K, N, BLOCK_K=BLOCK_K
        )
        return out