import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_partial_lse_kernel(
    A_ptr, B_ptr, bias_ptr,
    Mx_ptr, Sx_ptr,  # partial max and sum-of-exp per (m, n_block)
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mxm, stride_mxn,
    stride_sxm, stride_sxn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Mask invalid positions with -inf for LSE
    neg_inf = float('-inf')
    acc = tl.where(mask_m[:, None] & mask_n[None, :], acc, neg_inf)

    # Compute per-row partial max and sum(exp(x - max))
    row_max = tl.max(acc, axis=1)  # [BLOCK_M]
    # Handle case where all are -inf
    safe_max = tl.where(row_max == neg_inf, 0.0, row_max)
    exp_vals = tl.exp(acc - safe_max[:, None])
    exp_vals = tl.where(mask_m[:, None] & mask_n[None, :], exp_vals, 0.0)
    row_sum = tl.sum(exp_vals, axis=1)  # [BLOCK_M]

    # Store
    mx_ptrs = Mx_ptr + offs_m * stride_mxm + pid_n * stride_mxn
    sx_ptrs = Sx_ptr + offs_m * stride_sxm + pid_n * stride_sxn
    tl.store(mx_ptrs, row_max, mask=mask_m)
    tl.store(sx_ptrs, row_sum, mask=mask_m)


@triton.jit
def lse_reduce_act_kernel(
    Mx_ptr, Sx_ptr, out_ptr,
    M, NB,
    stride_mxm, stride_mxn,
    stride_sxm, stride_sxn,
    BLOCK_NB: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs = tl.arange(0, BLOCK_NB)
    mask = offs < NB
    neg_inf = float('-inf')
    mx = tl.load(Mx_ptr + pid * stride_mxm + offs * stride_mxn, mask=mask, other=neg_inf)
    sx = tl.load(Sx_ptr + pid * stride_sxm + offs * stride_sxn, mask=mask, other=0.0)

    # Combine: global_max = max(mx); sum = sum(sx * exp(mx - global_max)); lse = log(sum) + global_max
    gmax = tl.max(mx, axis=0)
    safe_gmax = tl.where(gmax == neg_inf, 0.0, gmax)
    adj = sx * tl.exp(mx - safe_gmax)
    adj = tl.where(mask, adj, 0.0)
    total = tl.sum(adj, axis=0)
    lse = tl.log(total) + safe_gmax

    # Apply two LeakyReLU (slope 0.01) - equivalent to one with slope 0.0001 for negatives
    x = lse
    x = tl.where(x >= 0.0, x, x * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

    # Apply GELU twice (exact form using erf)
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + pid, x)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight  # [out_features, in_features]
        b = self.linear.bias    # [out_features]
        # We compute A @ W^T where A=x [M, K], so B = W^T [K, N]
        M, K = x.shape
        N = self.out_features

        # B view: W is [N, K] row-major => W^T is [K, N] with strides (1, K)
        # We'll pass strides explicitly.
        A = x
        B = W  # treat as [K, N] via strides

        stride_am, stride_ak = A.stride(0), A.stride(1)
        # For W [N, K]: W[n, k] -> offset n*K + k. We want B[k, n] = W[n, k] => stride_bk=1, stride_bn=K
        stride_bk = 1
        stride_bn = K

        # Use a fixed BLOCK_N for partial storage layout: we need NB known after autotune.
        # Strategy: pick BLOCK_N=128 manually (skip autotune for predictability), or compute NB after launch.
        # We use autotune but query the chosen config from the kernel run isn't possible directly.
        # Solution: do our own fixed blocking for the partial buffer.
        BLOCK_N_FIXED = 128
        NB = (N + BLOCK_N_FIXED - 1) // BLOCK_N_FIXED

        Mx = torch.empty((M, NB), device=x.device, dtype=torch.float32)
        Sx = torch.empty((M, NB), device=x.device, dtype=torch.float32)

        # Use a non-autotuned launch with fixed config to match buffer layout
        BLOCK_M = 64
        BLOCK_N = BLOCK_N_FIXED
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

        # Call the kernel without autotune by using the .fn (bypassing autotuner with explicit config)
        _gemm_partial_lse_kernel_fixed[grid](
            A, B, b if b is not None else torch.zeros(N, device=x.device, dtype=x.dtype),
            Mx, Sx,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            Mx.stride(0), Mx.stride(1),
            Sx.stride(0), Sx.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_NB = _next_pow2(NB)
        lse_reduce_act_kernel[(M,)](
            Mx, Sx, out,
            M, NB,
            Mx.stride(0), Mx.stride(1),
            Sx.stride(0), Sx.stride(1),
            BLOCK_NB=BLOCK_NB,
        )
        return out


@triton.jit
def _gemm_partial_lse_kernel_fixed(
    A_ptr, B_ptr, bias_ptr,
    Mx_ptr, Sx_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mxm, stride_mxn,
    stride_sxm, stride_sxn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    neg_inf = float('-inf')
    acc = tl.where(mask_m[:, None] & mask_n[None, :], acc, neg_inf)

    row_max = tl.max(acc, axis=1)
    safe_max = tl.where(row_max == neg_inf, 0.0, row_max)
    exp_vals = tl.exp(acc - safe_max[:, None])
    exp_vals = tl.where(mask_m[:, None] & mask_n[None, :], exp_vals, 0.0)
    row_sum = tl.sum(exp_vals, axis=1)

    mx_ptrs = Mx_ptr + offs_m * stride_mxm + pid_n * stride_mxn
    sx_ptrs = Sx_ptr + offs_m * stride_sxm + pid_n * stride_sxn
    tl.store(mx_ptrs, row_max, mask=mask_m)
    tl.store(sx_ptrs, row_sum, mask=mask_m)