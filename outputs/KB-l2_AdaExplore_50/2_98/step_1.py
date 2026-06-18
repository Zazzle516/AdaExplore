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
        triton.Config({'BLOCK_P': 8, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 8, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_P': 16, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_P': 32, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['K', 'N_POOLED', 'POOL_K'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N_POOLED,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    POOL_K: tl.constexpr,
    SCALE: tl.constexpr,
    INV_POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)

    # Pooled output indices for this program
    p_idx = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)  # [BLOCK_P]
    p_mask = p_idx < N_POOLED

    # Linear output indices: BLOCK_P * POOL_K
    # n = p_idx * POOL_K + j, j in [0, POOL_K)
    # Shape [BLOCK_P, POOL_K]
    j = tl.arange(0, POOL_K)  # [POOL_K]
    n_idx = p_idx[:, None] * POOL_K + j[None, :]  # [BLOCK_P, POOL_K]
    n_mask = p_mask[:, None]

    # Accumulator
    acc = tl.zeros((BLOCK_P, POOL_K), dtype=tl.float32)

    # x row pointer: [K]
    x_row_ptr = x_ptr + pid_m * stride_xm

    # K-loop
    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        k_mask = k_idx < K
        # Load x: [BLOCK_K]
        x_vals = tl.load(x_row_ptr + k_idx * stride_xk, mask=k_mask, other=0.0)
        # Load W: [BLOCK_P*POOL_K, BLOCK_K]
        # W shape (N, K), stride_wn for N, stride_wk for K.
        # We want W[n_idx, k_idx]. Flatten n to [BLOCK_P, POOL_K].
        w_ptrs = w_ptr + n_idx[:, :, None] * stride_wn + k_idx[None, None, :] * stride_wk
        w_mask = n_mask[:, :, None] & k_mask[None, None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # Multiply and accumulate: acc[p, j] += sum_k x[k] * w[p,j,k]
        acc += tl.sum(w_vals * x_vals[None, None, :], axis=2)

    # Add bias: b indexed by n_idx
    b_vals = tl.load(b_ptr + n_idx, mask=n_mask, other=0.0)
    acc = acc + b_vals

    # Avg pool over j axis
    pooled = tl.sum(acc, axis=1) * INV_POOL  # [BLOCK_P]

    # GELU (exact, using erf)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))

    # Scale
    scaled = gelu * SCALE

    # Mask invalid pooled indices to -inf for max
    neg_inf = float('-inf')
    scaled = tl.where(p_mask, scaled, neg_inf)

    # Reduce max within this program
    block_max = tl.max(scaled, axis=0)

    # Atomic max into output[pid_m]
    tl.atomic_max(out_ptr + pid_m, block_max)


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
        N_POOLED = self.n_pooled
        POOL_K = self.pool_kernel_size

        # Output: per-row max, init to -inf
        out = torch.full((M,), float('-inf'), device=x.device, dtype=torch.float32)

        grid = lambda meta: (M, triton.cdiv(N_POOLED, meta['BLOCK_P']))

        fused_kernel[grid](
            x, w, b, out,
            M, K, N_POOLED,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            POOL_K=POOL_K,
            SCALE=self.scale_factor,
            INV_POOL=1.0 / POOL_K,
        )
        return out