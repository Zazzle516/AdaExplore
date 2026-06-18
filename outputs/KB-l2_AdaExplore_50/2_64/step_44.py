import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


GEMM_CONFIGS = [
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
]


@triton.autotune(configs=GEMM_CONFIGS, key=['M', 'N', 'K'])
@triton.jit
def gemm_partial_lse_kernel(
    A_ptr, B_ptr, bias_ptr,
    M_ptr, S_ptr,  # partial max [M, num_n_blocks], partial sumexp [M, num_n_blocks]
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mm, stride_mn,
    stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a = tl.load(a_ptrs + k * BLOCK_K * stride_ak,
                    mask=mask_m[:, None] & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs + k * BLOCK_K * stride_bk,
                    mask=(k_offs[:, None] < K) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Mask out-of-range columns to -inf
    NEG_INF = float('-inf')
    acc = tl.where(mask_n[None, :], acc, NEG_INF)

    # Per-row max over this N-tile
    row_max = tl.max(acc, axis=1)  # [BLOCK_M]
    # sum(exp(acc - row_max))
    row_max_safe = tl.where(row_max == NEG_INF, 0.0, row_max)
    sumexp = tl.sum(tl.exp(acc - row_max_safe[:, None]), axis=1)  # [BLOCK_M]

    # Store partial
    m_out = M_ptr + offs_m * stride_mm + pid_n * stride_mn
    s_out = S_ptr + offs_m * stride_sm + pid_n * stride_sn
    tl.store(m_out, row_max, mask=mask_m)
    tl.store(s_out, sumexp, mask=mask_m)


@triton.jit
def finalize_lse_kernel(
    M_ptr, S_ptr, OUT_ptr,
    M, NUM_BLOCKS,
    BLOCK_NB: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs = tl.arange(0, BLOCK_NB)
    mask = offs < NUM_BLOCKS
    NEG_INF = float('-inf')
    maxs = tl.load(M_ptr + pid * NUM_BLOCKS + offs, mask=mask, other=NEG_INF)
    sums = tl.load(S_ptr + pid * NUM_BLOCKS + offs, mask=mask, other=0.0)

    global_max = tl.max(maxs, axis=0)
    # sum_exp = sum(sums * exp(maxs - global_max))
    diff = maxs - global_max
    weights = tl.exp(diff)
    total = tl.sum(sums * weights * tl.where(mask, 1.0, 0.0), axis=0)
    lse = global_max + tl.log(total)

    # Apply leaky_relu twice (slope 0.01), then GELU twice
    slope = 0.01
    x = lse
    x = tl.where(x >= 0.0, x, x * slope)
    x = tl.where(x >= 0.0, x, x * slope)
    # GELU (exact via erf): 0.5*x*(1+erf(x/sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(OUT_ptr + pid, x)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous().cuda()  # [out, in]
        if self.linear.bias is not None:
            b = self.linear.bias.contiguous().cuda()
        else:
            b = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)

        M, K = x.shape
        N = self.out_features
        # B = W.T effectively; use strides
        # We treat A=x [M,K], B=W^T [K,N]: stride_bk = 1 (since W is [N,K] row-major, W^T element [k,n] = W[n,k])
        # W shape [N, K], strides (K, 1). W^T[k,n] = W[n,k] -> ptr = W_ptr + n*K + k. stride_bk=1, stride_bn=K.

        # Determine BLOCK_N from autotune; we need num_n_blocks. We'll pick a max possible and pad.
        # Simpler: do partial lse, then finalize. We need to know BLOCK_N after autotune, so use a wrapper grid.

        # Pre-allocate for worst-case BLOCK_N=64 -> max blocks. But autotune chooses; we don't know upfront.
        # Strategy: fix BLOCK_N for partial output indexing by making grid dimension based on meta.
        # We'll allocate partials based on a function of meta inside grid lambda. Allocate inside callback isn't possible.
        # Instead, allocate enough for the smallest BLOCK_N (64): num_blocks = ceil(N/64).
        # Then in kernel we use pid_n indexing the partial buffer with stride computed from actual num_n_blocks.

        # We'll do: pick BLOCK_N=128 fixed for partial layout; create non-autotuned version for simplicity.
        # Actually, autotune varies BLOCK_N. To handle this, allocate partials sized by max possible blocks (with smallest BLOCK_N=64).
        # But pid_n range matches actual config's num blocks. We pass actual num_n_blocks via stride.

        # Simpler: don't autotune. Use fixed config.
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32
        num_m_blocks = (M + BLOCK_M - 1) // BLOCK_M
        num_n_blocks = (N + BLOCK_N - 1) // BLOCK_N

        partial_max = torch.empty((M, num_n_blocks), device=x.device, dtype=torch.float32)
        partial_sum = torch.empty((M, num_n_blocks), device=x.device, dtype=torch.float32)

        grid = (num_m_blocks, num_n_blocks)
        _gemm_partial_lse_kernel_fixed[grid](
            x, W, b,
            partial_max, partial_sum,
            M, N, K,
            x.stride(0), x.stride(1),
            1, K,  # B = W^T
            partial_max.stride(0), partial_max.stride(1),
            partial_sum.stride(0), partial_sum.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        # next pow2 >= num_n_blocks
        BLOCK_NB = 1
        while BLOCK_NB < num_n_blocks:
            BLOCK_NB *= 2
        BLOCK_NB = max(BLOCK_NB, 1)

        finalize_lse_kernel[(M,)](
            partial_max, partial_sum, out,
            M, num_n_blocks,
            BLOCK_NB=BLOCK_NB,
        )

        return out


@triton.jit
def _gemm_partial_lse_kernel_fixed(
    A_ptr, B_ptr, bias_ptr,
    M_ptr, S_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mm, stride_mn,
    stride_sm, stride_sn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a = tl.load(a_ptrs + k * BLOCK_K * stride_ak,
                    mask=mask_m[:, None] & (k_offs[None, :] < K), other=0.0)
        b = tl.load(b_ptrs + k * BLOCK_K * stride_bk,
                    mask=(k_offs[:, None] < K) & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    NEG_INF = float('-inf')
    acc = tl.where(mask_n[None, :], acc, NEG_INF)

    row_max = tl.max(acc, axis=1)
    row_max_safe = tl.where(row_max == NEG_INF, 0.0, row_max)
    sumexp = tl.sum(tl.exp(acc - row_max_safe[:, None]), axis=1)

    m_out = M_ptr + offs_m * stride_mm + pid_n * stride_mn
    s_out = S_ptr + offs_m * stride_sm + pid_n * stride_sn
    tl.store(m_out, row_max, mask=mask_m)
    tl.store(s_out, sumexp, mask=mask_m)