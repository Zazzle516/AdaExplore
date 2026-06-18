import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_partial_lse_kernel(
    A_ptr, B_ptr, bias_ptr,
    max_partial_ptr, sumexp_partial_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mm, stride_mn,
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

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        mask_k = offs_k < k_remain
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # mask out-of-range to -inf so they don't affect max/sumexp
    neg_inf = float('-inf')
    valid = mask_m[:, None] & mask_n[None, :]
    acc = tl.where(valid, acc, neg_inf)

    # row-wise max over this N tile
    row_max = tl.max(acc, axis=1)  # [BLOCK_M]
    # sum exp(acc - row_max)
    shifted = acc - row_max[:, None]
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(valid, exp_vals, 0.0)
    row_sumexp = tl.sum(exp_vals, axis=1)  # [BLOCK_M]

    out_m_ptr = max_partial_ptr + offs_m * stride_mm + pid_n * stride_mn
    out_s_ptr = sumexp_partial_ptr + offs_m * stride_mm + pid_n * stride_mn
    tl.store(out_m_ptr, row_max, mask=mask_m)
    tl.store(out_s_ptr, row_sumexp, mask=mask_m)


@triton.jit
def finalize_lse_act_kernel(
    max_partial_ptr, sumexp_partial_ptr,
    out_ptr,
    M, NTILES,
    stride_mm, stride_mn,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs_t = tl.arange(0, BLOCK_T)
    mask_t = offs_t < NTILES

    m_ptrs = max_partial_ptr + pid * stride_mm + offs_t * stride_mn
    s_ptrs = sumexp_partial_ptr + pid * stride_mm + offs_t * stride_mn

    neg_inf = float('-inf')
    m_vals = tl.load(m_ptrs, mask=mask_t, other=neg_inf)
    s_vals = tl.load(s_ptrs, mask=mask_t, other=0.0)

    global_max = tl.max(m_vals, axis=0)
    # scale sumexp by exp(m_vals - global_max)
    scaled = s_vals * tl.exp(m_vals - global_max)
    scaled = tl.where(mask_t, scaled, 0.0)
    total = tl.sum(scaled, axis=0)
    lse = global_max + tl.log(total)

    # LeakyReLU twice with slope 0.01 (composed: slope = 0.0001 if x<0)
    x = lse
    x = tl.where(x >= 0, x, x * 0.01)
    x = tl.where(x >= 0, x, x * 0.01)

    # GELU twice (exact: 0.5*x*(1+erf(x/sqrt(2))))
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + pid, x)


def _next_pow2(x):
    p = 1
    while p < x:
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
        b = self.linear.bias
        if b is None:
            b = torch.zeros(self.out_features, device=x.device, dtype=x.dtype)
        else:
            b = b.contiguous()
        # B for gemm = W^T -> shape [in_features, out_features]
        Wt = W.t().contiguous()

        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        # Use fixed tile size for N to determine partial buffers
        # Use BLOCK_N from autotune varies; we'll allocate per pid_n max possible.
        # To make finalize stable, fix BLOCK_N for partials by using a wrapper grid based on chosen config.
        # Instead, use a fixed BLOCK_N for splits = 64 via a non-autotuned launch — but we want autotune.
        # Use a separate kernel call with autotune; grid uses meta. Allocate partials sized to max ntiles.

        # We'll pick a single config externally for predictable NTILES. Compromise: pick BLOCK_N=128.
        # Use a manual launch (no autotune) for predictability:
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32
        ntiles_n = (N + BLOCK_N - 1) // BLOCK_N
        ntiles_m = (M + BLOCK_M - 1) // BLOCK_M

        max_partial = torch.empty((M, ntiles_n), device=x.device, dtype=torch.float32)
        sumexp_partial = torch.empty((M, ntiles_n), device=x.device, dtype=torch.float32)

        grid = (ntiles_m, ntiles_n)
        _gemm_kernel_manual[grid](
            x, Wt, b,
            max_partial, sumexp_partial,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            max_partial.stride(0), max_partial.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=3,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_T = _next_pow2(ntiles_n)
        if BLOCK_T < 1:
            BLOCK_T = 1
        finalize_lse_act_kernel[(M,)](
            max_partial, sumexp_partial, out,
            M, ntiles_n,
            max_partial.stride(0), max_partial.stride(1),
            BLOCK_T=BLOCK_T,
        )
        return out


@triton.jit
def _gemm_kernel_manual(
    A_ptr, B_ptr, bias_ptr,
    max_partial_ptr, sumexp_partial_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mm, stride_mn,
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

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        mask_k = offs_k < k_remain
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    neg_inf = float('-inf')
    valid = mask_m[:, None] & mask_n[None, :]
    acc = tl.where(valid, acc, neg_inf)

    row_max = tl.max(acc, axis=1)
    shifted = acc - row_max[:, None]
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(valid, exp_vals, 0.0)
    row_sumexp = tl.sum(exp_vals, axis=1)

    out_m_ptr = max_partial_ptr + offs_m * stride_mm + pid_n * stride_mn
    out_s_ptr = sumexp_partial_ptr + offs_m * stride_mm + pid_n * stride_mn
    tl.store(out_m_ptr, row_max, mask=mask_m)
    tl.store(out_s_ptr, row_sumexp, mask=mask_m)