import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_pool_gelu_scale_partialmax_kernel(
    X_ptr, W_ptr, B_ptr, PartMax_ptr,
    M, N, K, P, NUM_PID_N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # BLOCK_N must be a multiple of POOL
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

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
    offs_n_full = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b_vals = tl.load(B_ptr + offs_n_full, mask=offs_n_full < N, other=0.0)
    acc += b_vals[None, :]

    # Mask invalid columns (out-of-bounds N) to 0 for pooling sum
    valid_n = offs_n_full < N
    acc = tl.where(valid_n[None, :], acc, 0.0)

    # Reshape acc to (BLOCK_M, BLOCK_N/POOL, POOL) and reduce sum -> divide by POOL = avg pool
    BLOCK_P: tl.constexpr = BLOCK_N // POOL
    acc_reshaped = tl.reshape(acc, (BLOCK_M, BLOCK_P, POOL))
    pooled = tl.sum(acc_reshaped, axis=2) * (1.0 / POOL)  # (BLOCK_M, BLOCK_P)

    # GELU (erf)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
    scaled = gelu * SCALE

    # Mask out pooled bins that come from invalid N region
    # A pooled bin index p (within tile) maps to columns [pid_n*BLOCK_N + p*POOL, pid_n*BLOCK_N + p*POOL + POOL)
    # Bin valid if start < P*POOL = N (since N is divisible by POOL).
    p_offs_global = pid_n * BLOCK_P + tl.arange(0, BLOCK_P)
    p_valid = p_offs_global < P
    scaled = tl.where(p_valid[None, :], scaled, -float('inf'))

    # Reduce over BLOCK_P -> per-row partial max for this N-tile
    part_max = tl.max(scaled, axis=1)  # (BLOCK_M,)

    # Store partial max to PartMax[m, pid_n]
    offs_m_full = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m_full < M
    out_ptrs = PartMax_ptr + offs_m_full * stride_pm + pid_n * stride_pn
    tl.store(out_ptrs, part_max, mask=m_mask)


@triton.jit
def row_max_kernel(
    PartMax_ptr, Out_ptr,
    M, NUM_TILES,
    stride_pm, stride_pn,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        offs = tl.arange(0, BLOCK_T)
        mask = offs < NUM_TILES
        vals = tl.load(PartMax_ptr + pid * stride_pm + offs * stride_pn,
                       mask=mask, other=-float('inf'))
        m = tl.max(vals, axis=0)
        tl.store(Out_ptr + pid, m)


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
        P = self.pooled_size

        # We need a partial-max buffer of shape (M, num_pid_n). num_pid_n depends on BLOCK_N (autotuned).
        # Allocate the maximum reasonable size once and pass strides.
        # The max BLOCK_N across configs is 256, so min num_pid_n is N/256. The min BLOCK_N is 128 -> max num_pid_n is N/128.
        # Allocate enough for BLOCK_N=64 to be safe.
        max_num_tiles = triton.cdiv(N, 64)
        partmax = torch.full((M, max_num_tiles), float('-inf'), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        gemm_bias_pool_gelu_scale_partialmax_kernel[grid](
            x, W, B, partmax,
            M, N, K, P, max_num_tiles,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partmax.stride(0), partmax.stride(1),
            SCALE=self.scale_factor,
            POOL=POOL,
        )

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        # Find smallest power of two >= max_num_tiles
        BLOCK_T = 1
        while BLOCK_T < max_num_tiles:
            BLOCK_T *= 2

        row_max_kernel[(M,)](
            partmax, out,
            M, max_num_tiles,
            partmax.stride(0), partmax.stride(1),
            BLOCK_T=BLOCK_T,
            num_warps=2,
        )

        return out