import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Fused kernel: per (batch_row, out_tile of pooled dim).
# For each row, compute matmul output for K consecutive output features (K = pool_kernel_size),
# average them, apply GELU, scale, and reduce max over the pooled dim.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_P': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 128, 'BLOCK_P': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=4, num_stages=4),
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
    BLOCK_M: tl.constexpr,   # batch tile
    BLOCK_K: tl.constexpr,   # IN_FEATURES tile
    BLOCK_P: tl.constexpr,   # number of pooled indices per inner chunk
):
    pid_b = tl.program_id(0)   # batch tile id

    row_start = pid_b * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    OF_TILE: tl.constexpr = BLOCK_P * POOL_K
    NEG_INF = float('-inf')

    # Running max per row
    row_max = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)

    # Iterate over pool-tile chunks
    for pchunk in range(0, POOLED_LEN, BLOCK_P):
        of_start = pchunk * POOL_K
        of_offs = of_start + tl.arange(0, OF_TILE)  # [OF_TILE]

        # Accumulator: [BLOCK_M, OF_TILE]
        acc = tl.zeros((BLOCK_M, OF_TILE), dtype=tl.float32)

        for k0 in range(0, IN_FEATURES, BLOCK_K):
            k_offs = k0 + tl.arange(0, BLOCK_K)

            # x: [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + rows[:, None] * IN_FEATURES + k_offs[None, :]
            x_tile = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)

            # wt: [BLOCK_K, OF_TILE] (contiguous along OF_TILE axis)
            wt_ptrs = wt_ptr + k_offs[:, None] * OUT_FEATURES + of_offs[None, :]
            w_tile = tl.load(wt_ptrs)

            acc += tl.dot(x_tile, w_tile)

        # Add bias
        bias = tl.load(b_ptr + of_offs)  # [OF_TILE]
        acc = acc + bias[None, :]

        # Reshape and avg pool
        acc = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL_K))
        avg = tl.sum(acc, axis=2) * (1.0 / POOL_K)  # [BLOCK_M, BLOCK_P]

        # GELU (exact)
        inv_sqrt2 = 0.70710678118654752440
        gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))

        # Scale
        scaled = gelu * SCALE  # [BLOCK_M, BLOCK_P]

        # Local max across BLOCK_P
        local_max = tl.max(scaled, axis=1)  # [BLOCK_M]

        row_max = tl.maximum(row_max, local_max)

    # Store final max
    out_ptrs = out_ptr + rows
    tl.store(out_ptrs, row_max, mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.in_features = in_features
        self.out_features = out_features
        self.pooled_len = out_features // pool_kernel_size
        self._wt_cache = None
        self._wt_version = None

    def _get_wt(self):
        w = self.matmul.weight
        if (self._wt_cache is None) or (self._wt_version != w._version) or (not self._wt_cache.is_cuda):
            wt = w.detach().t().contiguous().cuda()
            self._wt_cache = wt
            self._wt_version = w._version
        return self._wt_cache

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        wt = self._get_wt()
        b = self.matmul.bias.detach().contiguous().cuda()

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