import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64, 'BLOCK_P': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_P': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32, 'BLOCK_P': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 4}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 128, 'BLOCK_P': 8}, num_warps=8, num_stages=3),
    ],
    key=['IN_FEATURES', 'POOLED_LEN', 'POOL_K'],
)
@triton.jit
def fused_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
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
    pid_b = tl.program_id(0)
    pid_p = tl.program_id(1)

    row_start = pid_b * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < B

    of_start = pid_p * BLOCK_P * POOL_K
    OF_TILE: tl.constexpr = BLOCK_P * POOL_K
    of_offs = of_start + tl.arange(0, OF_TILE)

    acc = tl.zeros((BLOCK_M, OF_TILE), dtype=tl.float32)

    for k0 in range(0, IN_FEATURES, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IN_FEATURES

        x_ptrs = x_ptr + rows[:, None] * IN_FEATURES + k_offs[None, :]
        x_mask = row_mask[:, None] & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + of_offs[:, None] * IN_FEATURES + k_offs[None, :]
        w_mask = k_mask[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, tl.trans(w_tile))

    bias = tl.load(b_ptr + of_offs)
    acc = acc + bias[None, :]

    acc = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL_K))
    avg = tl.sum(acc, axis=2) / POOL_K

    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))

    scaled = gelu * SCALE

    local_max = tl.max(scaled, axis=1)

    out_ptrs = out_ptr + rows
    tl.atomic_max(out_ptrs, local_max, mask=row_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.in_features = in_features
        self.out_features = out_features
        self.pooled_len = out_features // pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        w = self.matmul.weight.contiguous()
        b = self.matmul.bias.contiguous()

        out = torch.full((B,), float('-inf'), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(B, meta['BLOCK_M']), triton.cdiv(self.pooled_len, meta['BLOCK_P']))

        fused_kernel[grid](
            x, w, b, out,
            B,
            self.in_features,
            self.out_features,
            self.pooled_len,
            self.pool_kernel_size,
            self.scale_factor,
        )
        return out