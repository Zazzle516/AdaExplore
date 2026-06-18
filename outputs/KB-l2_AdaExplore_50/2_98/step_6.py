import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'POOL_K'],
)
@triton.jit
def fused_gemm_pool_gelu_max_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    POOL_K: tl.constexpr,
    SCALE: tl.constexpr,
    INV_POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    # Persistent running max per row in registers
    NEG_INF = float('-inf')
    row_max = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)

    inv_sqrt2 = 0.70710678118654752440

    # Iterate over N tiles
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_remaining = K - k_start
            x = tl.load(x_ptrs, mask=(m_mask[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_remaining) & (n_mask[None, :]), other=0.0)
            acc += tl.dot(x, w, allow_tf32=True)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # bias
        b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + b[None, :]

        # avg pool: BLOCK_N -> BLOCK_N/POOL_K
        pooled = tl.reshape(acc, (BLOCK_M, BLOCK_N // POOL_K, POOL_K))
        pooled = tl.sum(pooled, axis=2) * INV_POOL  # [BLOCK_M, BLOCK_N/POOL_K]

        # GELU exact
        gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
        scaled = gelu * SCALE

        # mask invalid pooled positions
        p_idx = tl.arange(0, BLOCK_N // POOL_K)
        abs_p = (n_start // POOL_K) + p_idx
        p_mask = abs_p < (N // POOL_K)
        scaled = tl.where(p_mask[None, :], scaled, NEG_INF)

        # update running max
        tile_max = tl.max(scaled, axis=1)  # [BLOCK_M]
        row_max = tl.maximum(row_max, tile_max)

    # final write
    tl.store(out_ptr + offs_m, row_max, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Cache transposed weight [K, N]
        self._cached_wt = None
        self._cached_wptr = None

    def _get_wt(self):
        # If params haven't changed (data ptr same and not training), reuse
        wptr = self.weight.data_ptr()
        if (not self.training) and self._cached_wt is not None and self._cached_wptr == wptr:
            return self._cached_wt
        wt = self.weight.detach().t().contiguous()
        self._cached_wt = wt
        self._cached_wptr = wptr
        return wt

    def forward(self, x):
        x = x.contiguous().cuda()
        if self.training and self.weight.requires_grad:
            w_t = self.weight.t().contiguous()
        else:
            w_t = self._get_wt()

        b = self.bias.contiguous()
        M = x.shape[0]
        K = self.in_features
        N = self.out_features
        POOL_K = self.pool_kernel_size

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

        fused_gemm_pool_gelu_max_kernel[grid](
            x, w_t, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w_t.stride(0), w_t.stride(1),
            POOL_K=POOL_K,
            SCALE=self.scale_factor,
            INV_POOL=1.0 / POOL_K,
        )
        return out