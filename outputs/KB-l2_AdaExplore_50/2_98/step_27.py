import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Fused kernel using a pre-transposed weight [IN_FEATURES, OUT_FEATURES].
# Each program owns BLOCK_M rows and a slab of SPLIT pooled tiles, scanning the
# pooled axis in-register and accumulating a running max. A final atomic_max
# combines the SPLIT partials per row.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32, 'BLOCK_P': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 4}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'POOLED_LEN', 'POOL_K', 'SPLIT'],
)
@triton.jit
def fused_kernel(
    x_ptr,         # [B, IN_FEATURES]
    wt_ptr,        # [IN_FEATURES, OUT_FEATURES] (transposed weight)
    b_ptr,         # [OUT_FEATURES]
    out_ptr,       # [B]
    B,
    IN_FEATURES: tl.constexpr,
    OUT_FEATURES: tl.constexpr,
    POOLED_LEN: tl.constexpr,
    POOL_K: tl.constexpr,
    SCALE: tl.constexpr,
    SPLIT: tl.constexpr,        # number of programs across pooled dim per row tile
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    row_start = pid_b * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    PK_TOTAL: tl.constexpr = BLOCK_P * POOL_K
    NUM_PTILES: tl.constexpr = POOLED_LEN // BLOCK_P
    PTILES_PER_SPLIT: tl.constexpr = NUM_PTILES // SPLIT

    pt_start = pid_s * PTILES_PER_SPLIT

    running_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)

    k_range = tl.arange(0, BLOCK_K)
    p_range = tl.arange(0, PK_TOTAL)

    for pt_idx in range(0, PTILES_PER_SPLIT):
        pt = pt_start + pt_idx
        of_start = pt * PK_TOTAL
        of_offs = of_start + p_range  # [PK_TOTAL]

        acc = tl.zeros((BLOCK_M, PK_TOTAL), dtype=tl.float32)

        for k0 in range(0, IN_FEATURES, BLOCK_K):
            k_offs = k0 + k_range
            # x: [BLOCK_M, BLOCK_K]
            x_ptrs = x_ptr + rows[:, None] * IN_FEATURES + k_offs[None, :]
            x_tile = tl.load(x_ptrs, mask=row_mask[:, None], other=0.0)
            # wt: [BLOCK_K, PK_TOTAL] — contiguous along PK_TOTAL (inner dim)
            wt_ptrs = wt_ptr + k_offs[:, None] * OUT_FEATURES + of_offs[None, :]
            w_tile = tl.load(wt_ptrs)
            acc += tl.dot(x_tile, w_tile)

        bias = tl.load(b_ptr + of_offs)
        acc = acc + bias[None, :]
        acc = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL_K))
        avg = tl.sum(acc, axis=2) * (1.0 / POOL_K)
        inv_sqrt2 = 0.70710678118654752440
        gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))
        scaled = gelu * SCALE
        local_max = tl.max(scaled, axis=1)
        running_max = tl.maximum(running_max, local_max)

    out_ptrs = out_ptr + rows
    if SPLIT == 1:
        tl.store(out_ptrs, running_max, mask=row_mask)
    else:
        tl.atomic_max(out_ptrs, running_max, mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.in_features = in_features
        self.out_features = out_features
        self.pooled_len = out_features // pool_kernel_size
        # Pre-transpose weight once so the inner GEMM load is contiguous along N.
        wt = self.matmul.weight.detach().t().contiguous()
        self.register_buffer('weight_t', wt, persistent=False)
        # Use SPLIT=1 to avoid atomic_max epilogue. With BLOCK_M=64 and B=1024
        # we get 16 row-tiles which is enough work per SM along with the pooled loop.
        self.SPLIT = 1

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        wt = self.weight_t
        b = self.matmul.bias.contiguous()

        if self.SPLIT > 1:
            out = torch.full((B,), float('-inf'), device=x.device, dtype=torch.float32)
        else:
            out = torch.empty((B,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']), self.SPLIT)

        fused_kernel[grid](
            x, wt, b, out,
            B,
            self.in_features,
            self.out_features,
            self.pooled_len,
            self.pool_kernel_size,
            self.scale_factor,
            self.SPLIT,
        )
        return out