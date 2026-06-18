import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K', 'POOL'],
)
@triton.jit
def fused_gemm_pool_gelu_scale_partialmax_kernel(
    X_ptr, W_ptr, B_ptr, PartialMax_ptr,
    M, N, K, NTILES,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D grid: (pid_m, pid_n)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    am_mask = offs_am < M
    bn_mask = offs_bn < N

    a_rows = tl.where(am_mask, offs_am, 0)
    b_cols = tl.where(bn_mask, offs_bn, 0)

    x_ptrs = X_ptr + (a_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (b_cols[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # bias add
    b_vals = tl.load(B_ptr + offs_bn, mask=bn_mask, other=0.0)
    acc += b_vals[None, :]

    # Now perform pooling along the N dimension with kernel POOL.
    # BLOCK_N must be divisible by POOL.
    # Reshape acc (BLOCK_M, BLOCK_N) -> (BLOCK_M, BLOCK_N/POOL, POOL), reduce last axis.
    BLOCK_P: tl.constexpr = BLOCK_N // POOL
    acc_3d = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL))
    pooled = tl.sum(acc_3d, axis=2) * (1.0 / POOL)  # (BLOCK_M, BLOCK_P)

    # GELU(pooled) * SCALE
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
    scaled = gelu * SCALE  # (BLOCK_M, BLOCK_P)

    # Mask invalid pooled lanes (those outside N when accounting for padding)
    # A pooled bin p is valid iff (pid_n*BLOCK_N + p*POOL + POOL-1) < N, i.e. all POOL elements valid.
    p_offs = pid_n * BLOCK_P + tl.arange(0, BLOCK_P)  # global pooled idx
    P_total = N // POOL
    p_valid = p_offs < P_total
    NEG_INF = float('-inf')
    scaled = tl.where((am_mask[:, None]) & (p_valid[None, :]), scaled, NEG_INF)

    # row-wise max within tile: (BLOCK_M,)
    tile_max = tl.max(scaled, axis=1)

    # Write partial max to PartialMax[pid_m*BLOCK_M + i, pid_n]
    out_row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = PartialMax_ptr + out_row_idx * NTILES + pid_n
    tl.store(out_ptrs, tile_max, mask=am_mask)


@triton.jit
def row_max_kernel(
    Pmax_ptr, Out_ptr,
    M, NTILES,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs = tl.arange(0, BLOCK_T)
    mask = offs < NTILES
    NEG_INF = float('-inf')
    vals = tl.load(Pmax_ptr + pid * NTILES + offs, mask=mask, other=NEG_INF)
    m = tl.max(vals, axis=0)
    tl.store(Out_ptr + pid, m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = int(pool_kernel_size)
        self.scale_factor = float(scale_factor)

        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())

        assert out_features % self.pool_kernel_size == 0
        self.pooled_size = out_features // self.pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        W = self.weight
        B = self.bias

        M, K = x.shape
        N = self.out_features
        POOL = self.pool_kernel_size

        # We need BLOCK_N divisible by POOL. Triton autotune will pick BLOCK_N at runtime,
        # but we need NTILES to launch the second kernel and allocate buffer. So we can't
        # know BLOCK_N up-front. Strategy: launch with a fixed BLOCK_N choice (skip autotune
        # for tiles dim) - instead we manually pick configs whose BLOCK_N is divisible by POOL=16.
        # All configs above have BLOCK_N in {64,128,256}, all divisible by 16.

        # We need to know NTILES = ceil(N / BLOCK_N) before launch so we can allocate
        # PartialMax. Use a wrapper that uses a chosen BLOCK_N. Simplest: pick BLOCK_N=128
        # and don't autotune BLOCK_N — but we want autotune for perf.
        # Workaround: allocate PartialMax assuming worst case (smallest BLOCK_N=64),
        # NTILES_max = ceil(N/64). The kernel writes only its tiles, which are determined
        # at autotune time. We need stride matching. Solution: pre-run a tiny call to fix
        # config, OR pick BLOCK_N statically.
        #
        # We'll go with a static BLOCK_N=128 (good for N=8192) and hand-tune num_warps/stages
        # via a small autotune over BLOCK_M and BLOCK_K only.

        # For simplicity, use a fixed BLOCK_N. Reimplement with a non-autotuned launch:
        BLOCK_N = 128
        BLOCK_M = 128
        BLOCK_K = 32
        NTILES = (N + BLOCK_N - 1) // BLOCK_N

        partial_max = torch.empty((M, NTILES), device=x.device, dtype=torch.float32)

        grid = ((M + BLOCK_M - 1) // BLOCK_M, NTILES)

        _fused_kernel_static[grid](
            x, W, B, partial_max,
            M, N, K, NTILES,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SCALE=self.scale_factor,
            POOL=POOL,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        # Find next pow2 >= NTILES
        BLOCK_T = 1
        while BLOCK_T < NTILES:
            BLOCK_T *= 2
        if BLOCK_T < 8:
            BLOCK_T = 8

        row_max_kernel[(M,)](
            partial_max, out,
            M, NTILES,
            BLOCK_T=BLOCK_T,
            num_warps=2,
            num_stages=2,
        )

        return out


@triton.jit
def _fused_kernel_static(
    X_ptr, W_ptr, B_ptr, PartialMax_ptr,
    M, N, K, NTILES,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    am_mask = offs_am < M
    bn_mask = offs_bn < N

    a_rows = tl.where(am_mask, offs_am, 0)
    b_cols = tl.where(bn_mask, offs_bn, 0)

    x_ptrs = X_ptr + (a_rows[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (b_cols[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b_vals = tl.load(B_ptr + offs_bn, mask=bn_mask, other=0.0)
    acc += b_vals[None, :]

    BLOCK_P: tl.constexpr = BLOCK_N // POOL
    acc_3d = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL))
    pooled = tl.sum(acc_3d, axis=2) * (1.0 / POOL)

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
    scaled = gelu * SCALE

    p_offs = pid_n * BLOCK_P + tl.arange(0, BLOCK_P)
    P_total = N // POOL
    p_valid = p_offs < P_total
    NEG_INF = float('-inf')
    scaled = tl.where(am_mask[:, None] & p_valid[None, :], scaled, NEG_INF)

    tile_max = tl.max(scaled, axis=1)

    out_row_idx = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_ptrs = PartialMax_ptr + out_row_idx * NTILES + pid_n
    tl.store(out_ptrs, tile_max, mask=am_mask)