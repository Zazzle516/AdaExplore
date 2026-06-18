import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'POOL'],
)
@triton.jit
def fused_gemm_pool_gelu_max_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program computes BLOCK_M rows × BLOCK_N output features, then
    # reduces pool/gelu/scale/max contribution for these N features and
    # atomically updates the per-row running max.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    offs_am = offs_m % M
    offs_bn = offs_n % N

    x_ptrs = X_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_bn[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b_vals = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b_vals[None, :]

    # Mask out-of-range N columns so they don't affect pooling/max
    n_valid = offs_n < N
    acc = tl.where(n_valid[None, :], acc, 0.0)

    # Pool: reshape BLOCK_N -> (BLOCK_N/POOL, POOL), average along POOL
    # BLOCK_N is a multiple of POOL.
    BLOCK_P: tl.constexpr = BLOCK_N // POOL
    acc_r = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL))
    pooled = tl.sum(acc_r, axis=2) * (1.0 / POOL)  # (BLOCK_M, BLOCK_P)

    # GELU (erf form)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
    scaled = gelu * SCALE

    # Mask invalid pooled bins (those whose entire window was out-of-range N)
    # A pooled bin p covers N indices [pid_n*BLOCK_N + p*POOL, +POOL)
    # If pid_n*BLOCK_N + p*POOL >= N, bin invalid.
    p_idx = tl.arange(0, BLOCK_P)
    bin_start = pid_n * BLOCK_N + p_idx * POOL
    bin_valid = bin_start < N
    scaled = tl.where(bin_valid[None, :], scaled, -float('inf'))

    # Reduce max across pooled bins -> (BLOCK_M,)
    tile_max = tl.max(scaled, axis=1)

    # Mask invalid rows
    m_valid = offs_m < M
    tile_max = tl.where(m_valid, tile_max, -float('inf'))

    # Atomic max into Out_ptr[offs_m]
    tl.atomic_max(Out_ptr + offs_m, tile_max, mask=m_valid)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)

        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())

        assert out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size"
        self.pooled_size = out_features // pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.weight.contiguous()
        B = self.bias.contiguous()

        M, K = x.shape
        N = self.out_features
        POOL = self.pool_kernel_size

        # Initialize output to -inf for atomic max
        out = torch.full((M,), float('-inf'), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )

        fused_gemm_pool_gelu_max_kernel[grid](
            x, W, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SCALE=self.scale_factor,
            POOL=POOL,
        )

        return out