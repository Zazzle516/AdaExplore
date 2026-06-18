import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Fused: matmul (x @ W^T + b), avg_pool1d (kernel=pool_k), GELU, scale, max-reduce
# Strategy: one program per (batch_row, group of pooled outputs).
# We compute groups of `BLOCK_P` pooled outputs => BLOCK_P * pool_k linear outputs.
# K reduction is tiled over BLOCK_K.
# After the K-loop, apply bias, avg-pool (mean over pool_k), GELU, scale,
# then take max over the BLOCK_P pooled outputs in this program, and atomic-max
# into the per-row output.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_P': 8, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 8, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 16, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 16, 'BLOCK_K': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_P': 16, 'BLOCK_K': 256}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_P': 8, 'BLOCK_K': 256}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_P': 4, 'BLOCK_K': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_P': 8, 'BLOCK_K': 512}, num_warps=8, num_stages=3),
    ],
    key=['K', 'N_POOLED', 'POOL_K'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N, N_POOLED,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    POOL_K: tl.constexpr,
    SCALE: tl.constexpr,
    INV_POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    # x row pointer: [K]
    x_row_ptr = x_ptr + pid_m * stride_xm

    offs_k = tl.arange(0, BLOCK_K)
    # Treat N tile as BLOCK_P * POOL_K
    offs_n_tile = tl.arange(0, BLOCK_P * POOL_K)

    running_max = float('-inf')

    for p_start in range(0, N_POOLED, BLOCK_P):
        # Linear output indices for this tile
        n_offs = p_start * POOL_K + offs_n_tile  # [BLOCK_N]
        n_mask = n_offs < N

        acc = tl.zeros((1, BLOCK_P * POOL_K), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            k_mask = k_offs < K
            # x: [1, BLOCK_K]
            x_vals = tl.load(x_row_ptr + k_offs * stride_xk, mask=k_mask, other=0.0)
            x_2d = x_vals[None, :]
            # w: [BLOCK_K, BLOCK_N]
            w_ptrs = w_ptr + n_offs[None, :] * stride_wn + k_offs[:, None] * stride_wk
            w_mask = n_mask[None, :] & k_mask[:, None]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x_2d, w_vals)

        # Flatten to [BLOCK_N]
        acc1d = tl.reshape(acc, (BLOCK_P * POOL_K,))
        # Bias
        b_vals = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
        acc1d = acc1d + b_vals
        # Zero out invalid lanes (so they contribute 0 to pool sum; invalid pooled rows masked later)
        acc1d = tl.where(n_mask, acc1d, 0.0)

        # Reshape to [BLOCK_P, POOL_K] and avg-pool
        acc2d = tl.reshape(acc1d, (BLOCK_P, POOL_K))
        pooled = tl.sum(acc2d, axis=1) * INV_POOL  # [BLOCK_P]

        # GELU (exact)
        inv_sqrt2 = 0.70710678118654752440
        gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
        scaled = gelu * SCALE

        # Mask invalid pooled positions
        p_idx = p_start + tl.arange(0, BLOCK_P)
        p_mask = p_idx < N_POOLED
        scaled = tl.where(p_mask, scaled, float('-inf'))

        tile_max = tl.max(scaled, axis=0)
        running_max = tl.maximum(running_max, tile_max)

    tl.store(out_ptr + pid_m, running_max)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        # Match nn.Linear init
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

        # Number of pooled outputs
        self.n_pooled = out_features // pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.weight.contiguous()
        b = self.bias.contiguous()
        M = x.shape[0]
        K = self.in_features
        N = self.out_features
        N_POOLED = self.n_pooled
        POOL_K = self.pool_kernel_size

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = (M,)

        fused_kernel[grid](
            x, w, b, out,
            M, K, N, N_POOLED,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            POOL_K=POOL_K,
            SCALE=self.scale_factor,
            INV_POOL=1.0 / POOL_K,
        )
        return out