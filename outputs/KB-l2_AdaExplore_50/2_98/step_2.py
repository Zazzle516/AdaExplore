import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'POOL_K'],
)
@triton.jit
def gemm_pool_gelu_max_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K, NUM_N_TILES,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pm, stride_pt,
    POOL_K: tl.constexpr,
    SCALE: tl.constexpr,
    INV_POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M
    n_mask = offs_n < N

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is laid out as [K, N] so K stride on rows, N stride on cols
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

    # avg pool: reshape BLOCK_N into (BLOCK_N/POOL_K, POOL_K), mean
    # BLOCK_N must be multiple of POOL_K
    pooled = tl.reshape(acc, (BLOCK_M, BLOCK_N // POOL_K, POOL_K))
    pooled = tl.sum(pooled, axis=2) * INV_POOL  # [BLOCK_M, BLOCK_N/POOL_K]

    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
    scaled = gelu * SCALE

    # Mask invalid pooled positions
    # The pooled index p corresponds to original n = p*POOL_K to p*POOL_K+POOL_K-1
    # in absolute coordinates: pid_n*BLOCK_N + p*POOL_K
    p_idx = tl.arange(0, BLOCK_N // POOL_K)
    abs_p = pid_n * (BLOCK_N // POOL_K) + p_idx
    p_mask = abs_p < (N // POOL_K)
    neg_inf = float('-inf')
    scaled = tl.where(p_mask[None, :], scaled, neg_inf)

    # max along the N tile
    block_max = tl.max(scaled, axis=1)  # [BLOCK_M]

    # write to partial[m, pid_n]
    out_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pt
    tl.store(out_ptrs, block_max, mask=m_mask)


@triton.jit
def reduce_max_kernel(
    partial_ptr, out_ptr,
    M, NUM_TILES,
    stride_pm, stride_pt,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    offs = tl.arange(0, BLOCK_T)
    mask = offs < NUM_TILES
    vals = tl.load(partial_ptr + pid_m * stride_pm + offs * stride_pt,
                   mask=mask, other=float('-inf'))
    m = tl.max(vals, axis=0)
    tl.store(out_ptr + pid_m, m)


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

        # Pre-transpose weight to [K, N] for coalesced dot loads
        self.register_buffer('weight_t', self.weight.detach().t().contiguous(), persistent=False)

    def forward(self, x):
        x = x.contiguous().cuda()
        # Use the pre-transposed weight; but since training could update weight,
        # rebuild weight_t each forward. For inference this is one transpose.
        if (not self.weight.requires_grad) or (not self.training):
            # rebuild only if shape changed (it doesn't); reuse cached
            if self.weight_t.data_ptr() == 0 or self.weight_t.shape != (self.in_features, self.out_features):
                self.weight_t = self.weight.detach().t().contiguous()
            w_t = self.weight_t
        else:
            w_t = self.weight.t().contiguous()

        b = self.bias.contiguous()
        M = x.shape[0]
        K = self.in_features
        N = self.out_features
        POOL_K = self.pool_kernel_size

        # We need BLOCK_N to be divisible by POOL_K. All configs use BLOCK_N in {64,128,256}, POOL_K=16 -> ok.

        # Compute number of N tiles based on autotune-chosen BLOCK_N
        # Use a partial buffer big enough for worst case (smallest BLOCK_N)
        # Actually we need exact NUM_N_TILES from chosen config; allocate inside grid lambda after autotune?
        # Simpler: allocate based on max possible num tiles; but we need exact size for reduce.
        # Solution: do a two-step - first determine BLOCK_N via a fixed choice OR allocate worst-case and pass.
        # Use a lambda that allocates inside. But we can't allocate inside grid lambda meaningfully.
        # Instead: pre-pick a partial buffer assuming smallest BLOCK_N=64 -> num_tiles=N/64
        # Then within kernel each program writes to its pid_n slot (sparse if BLOCK_N is larger).
        # Then reduce reads NUM_TILES = used count.
        # Better: do separate path - run autotune once on a probe to get config? Complexity.
        # Simplest robust: pick a fixed BLOCK_N strategy: allocate worst case = N // 64, fill with -inf, kernel writes pid_n indexed. But pid_n range depends on BLOCK_N chosen at autotune time.
        # We'll use grid lambda with meta to compute num_tiles, but we need the partial allocated to that size.
        # Solution: allocate partial dynamically inside grid via closure? grid lambda runs at launch.
        # We'll allocate the partial buffer outside, sized to max possible (N // smallest_BLOCK_N = N // 64),
        # initialize to -inf, and reduce over actual num tiles set after launch.
        # But we don't know actual num_tiles after launch from outside cleanly... unless we read meta.
        # Workaround: use a pre_hook? Simpler: run a small autotune-resolving call, or just fix BLOCK_N.

        # Pragmatic fix: don't rely on autotune meta from python. Use a wrapper that does autotune internally
        # and reduce knowing the post-hoc BLOCK_N.
        # We'll use a different approach: allocate partial of max size, kernel writes to pid_n,
        # reducer reduces over actual num tiles which we read from kernel's chosen BLOCK_N via best_config.

        # Allocate worst-case partial
        max_tiles = (N + 63) // 64
        partial = torch.full((M, max_tiles), float('-inf'), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        gemm_pool_gelu_max_kernel[grid](
            x, w_t, b, partial,
            M, N, K, max_tiles,
            x.stride(0), x.stride(1),
            w_t.stride(0), w_t.stride(1),
            partial.stride(0), partial.stride(1),
            POOL_K=POOL_K,
            SCALE=self.scale_factor,
            INV_POOL=1.0 / POOL_K,
        )

        # Determine actual NUM_TILES from chosen BLOCK_N
        best_cfg = gemm_pool_gelu_max_kernel.best_config
        BLOCK_N_chosen = best_cfg.kwargs['BLOCK_N']
        num_tiles = (N + BLOCK_N_chosen - 1) // BLOCK_N_chosen

        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_T = triton.next_power_of_2(max_tiles)
        if BLOCK_T < 16:
            BLOCK_T = 16

        reduce_max_kernel[(M,)](
            partial, out,
            M, max_tiles,
            partial.stride(0), partial.stride(1),
            BLOCK_T=BLOCK_T,
        )
        return out