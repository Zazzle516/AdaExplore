import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowmax_kernel(
    X_ptr, WT_ptr, B_ptr, ROWMAX_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # X: (M, K) row-major; WT: (K, N) so the inner contiguous load is along N.
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    wt_ptrs = WT_ptr + offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        mask_k = offs_k < k_remain
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(wt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        wt_ptrs += BLOCK_K * stride_wtk

    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = tl.where(mask_n[None, :], acc, -float('inf'))
    row_max = tl.max(acc, axis=1)

    # Atomically reduce row-max across all N-tiles into a single (M,) buffer.
    tl.atomic_max(ROWMAX_ptr + offs_m, row_max, mask=mask_m)


@triton.jit
def gelu_epilogue_kernel(
    ROWMAX_ptr, OUT_ptr, M,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    x = tl.load(ROWMAX_ptr + offs, mask=mask, other=0.0)
    # (x - mean(x over dim=1)) where the column is size 1 -> always 0.
    y = x - x
    # GELU(0) = 0, but compute it explicitly to honor the op.
    # gelu(z) = 0.5 * z * (1 + erf(z / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    out = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    tl.store(OUT_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.in_features = in_features
        self.out_features = out_features
        # Pre-transpose W once: (K, N), contiguous along N for the inner load.
        with torch.no_grad():
            wt = self.gemm.weight.detach().t().contiguous().cuda()
        self.register_buffer('_wt', wt)

    def forward(self, x):
        x = x.contiguous().cuda()
        WT = self._wt
        B = self.gemm.bias.contiguous().cuda()

        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        if self.max_dim == 1:
            row_max = torch.full((M,), float('-inf'), device=x.device, dtype=torch.float32)

            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            gemm_rowmax_kernel[grid](
                x, WT, B, row_max,
                M, N, K,
                x.stride(0), x.stride(1),
                WT.stride(0), WT.stride(1),
            )

            out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
            BLOCK = 1024
            grid2 = (triton.cdiv(M, BLOCK),)
            gelu_epilogue_kernel[grid2](row_max, out, M, BLOCK=BLOCK, num_warps=4)
            return out
        else:
            y = torch.nn.functional.linear(x, self.gemm.weight, self.gemm.bias)
            y = torch.max(y, dim=self.max_dim, keepdim=True).values
            y = y - y.mean(dim=1, keepdim=True)
            return torch.nn.functional.gelu(y)