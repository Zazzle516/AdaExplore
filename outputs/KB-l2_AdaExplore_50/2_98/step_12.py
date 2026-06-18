import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Fused kernel: one program per batch tile.
# For each batch tile, loop over pooled groups; compute matmul tile, avg-pool,
# GELU, scale, and accumulate the running max in registers. No atomics needed.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64,  'BLOCK_P': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 256, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64,  'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 4}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 256, 'BLOCK_P': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64,  'BLOCK_P': 16}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'POOLED_LEN', 'POOL_K'],
)
@triton.jit
def fused_kernel(
    x_ptr,         # [B, IN_FEATURES]
    wt_ptr,        # [IN_FEATURES, OUT_FEATURES] (pre-transposed)
    b_ptr,         # [OUT_FEATURES]
    out_ptr,       # [B]
    B,
    IN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    POOLED_LEN: tl.constexpr,
    POOL_K: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    NF: tl.constexpr = BLOCK_P * POOL_K

    # running max init to -inf
    running_max = tl.zeros((BLOCK_M,), dtype=tl.float32) - float('inf')

    inv_sqrt2 = 0.70710678118654752440
    inv_pool_k = 1.0 / POOL_K

    for pg in range(0, POOLED_LEN, BLOCK_P):
        of_start = pg * POOL_K
        of_offs = of_start + tl.arange(0, NF)

        acc = tl.zeros((BLOCK_M, NF), dtype=tl.float32)

        for k0 in range(0, IN_FEATURES, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)

            x_ptrs = x_ptr + rows[:, None] * IN_FEATURES + k_offs[None, :]
            x_tile = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)

            # wt is [IN_FEATURES, OUT_FEATURES], contiguous in OUT_FEATURES
            wt_ptrs = wt_ptr + k_offs[:, None] * OUT_FEATURES + of_offs[None, :]
            w_tile = tl.load(wt_ptrs)

            acc += tl.dot(x_tile, w_tile)

        bias = tl.load(b_ptr + of_offs)
        acc = acc + bias[None, :]

        acc3 = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL_K))
        avg = tl.sum(acc3, axis=2) * inv_pool_k

        gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))
        scaled = gelu * SCALE

        local_max = tl.max(scaled, axis=1)
        running_max = tl.maximum(running_max, local_max)

    tl.store(out_ptr + rows, running_max, mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.in_features = in_features
        self.out_features = out_features
        self.pooled_len = out_features // pool_kernel_size
        # Pre-transposed weight cache
        self._wt_cache = None

    def _get_wt(self):
        w = self.matmul.weight
        if (self._wt_cache is None
                or self._wt_cache.device != w.device
                or self._wt_cache.dtype != w.dtype):
            self._wt_cache = w.t().contiguous()
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        wt = self._get_wt()
        b = self.matmul.bias.contiguous()

        out = torch.empty((B,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']),)

        fused_kernel[grid](
            x, wt, b, out,
            B,
            self.in_features,
            self.out_features,
            self.pooled_len,
            self.pool_kernel_size,
            self.scale_factor,
        )
        return out