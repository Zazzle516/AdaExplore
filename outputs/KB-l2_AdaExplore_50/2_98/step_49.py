import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'POOL'],
)
@triton.jit
def fused_gemm_pool_gelu_max_kernel(
    X_ptr, Wt_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per row-tile; loops over N accumulating running max.
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_am = offs_m % M
    m_valid = offs_m < M

    BLOCK_P: tl.constexpr = BLOCK_N // POOL
    inv_pool: tl.constexpr = 1.0 / POOL
    inv_sqrt2: tl.constexpr = 0.7071067811865475

    running_max = tl.full((BLOCK_M,), -float('inf'), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    for tile_n in range(0, num_n_tiles):
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_bn = offs_n % N

        x_ptrs = X_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = Wt_ptr + (offs_k[:, None] * stride_wtk + offs_bn[None, :] * stride_wtn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wtk

        # Add bias
        b_vals = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += b_vals[None, :]

        n_valid = offs_n < N
        acc = tl.where(n_valid[None, :], acc, 0.0)

        # Pool
        acc_r = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL))
        pooled = tl.sum(acc_r, axis=2) * inv_pool

        # GELU * scale
        gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
        scaled = gelu * SCALE

        # Mask invalid pooled bins
        p_idx = tl.arange(0, BLOCK_P)
        bin_start = tile_n * BLOCK_N + p_idx * POOL
        bin_valid = bin_start < N
        scaled = tl.where(bin_valid[None, :], scaled, -float('inf'))

        tile_max = tl.max(scaled, axis=1)
        running_max = tl.maximum(running_max, tile_max)

    running_max = tl.where(m_valid, running_max, -float('inf'))
    tl.store(Out_ptr + offs_m, running_max, mask=m_valid)


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

        # Pre-transpose: store as [K, N] contiguous so inner load is contiguous in N.
        with torch.no_grad():
            wt = lin.weight.detach().t().contiguous()  # [K, N]
        self.register_buffer('weight_t', wt, persistent=False)

        assert out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size"
        self.pooled_size = out_features // pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        # Use up-to-date weight_t (built once); rebuild lazily if weight has changed device/values
        if self.weight_t.device != x.device or self.weight_t.shape[0] != self.in_features:
            self.weight_t = self.weight.detach().t().contiguous().to(x.device)
        Wt = self.weight_t
        B = self.bias.contiguous()

        M, K = x.shape
        N = self.out_features
        POOL = self.pool_kernel_size

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)

        fused_gemm_pool_gelu_max_kernel[grid](
            x, Wt, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            SCALE=self.scale_factor,
            POOL=POOL,
        )

        return out