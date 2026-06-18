import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K', 'POOL'],
)
@triton.jit
def fused_gemm_pool_gelu_max_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    SCALE: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # GROUP_M swizzle: improves L2 reuse of W tiles across rows.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

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

    # Store tile max to scratch buffer at (offs_m, pid_n)
    out_ptrs = Out_ptr + offs_m * stride_om + pid_n * stride_on
    tl.store(out_ptrs, tile_max, mask=m_valid)


@triton.jit
def _row_max_kernel(
    In_ptr, Out_ptr,
    M, NT,
    stride_im, stride_in,
    BLOCK_NT: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        offs = tl.arange(0, BLOCK_NT)
        mask = offs < NT
        vals = tl.load(In_ptr + pid * stride_im + offs * stride_in, mask=mask, other=-float('inf'))
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

        # Scratch buffer for per-tile maxes
        # We don't know BLOCK_N until autotuning picks it, so allocate worst-case
        # based on smallest BLOCK_N in configs (128) -> NT_max = N // 128
        NT_max = triton.cdiv(N, 128)
        scratch = torch.full((M, NT_max), float('-inf'), device=x.device, dtype=torch.float32)
        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
        )

        fused_gemm_pool_gelu_max_kernel[grid](
            x, W, B, scratch,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            scratch.stride(0), scratch.stride(1),
            SCALE=self.scale_factor,
            POOL=POOL,
        )

        # Second-stage reduction: row max across NT_max columns
        # Find smallest power of two >= NT_max
        BLOCK_NT = 1
        while BLOCK_NT < NT_max:
            BLOCK_NT *= 2
        _row_max_kernel[(M,)](
            scratch, out,
            M, NT_max,
            scratch.stride(0), scratch.stride(1),
            BLOCK_NT=BLOCK_NT,
        )

        return out