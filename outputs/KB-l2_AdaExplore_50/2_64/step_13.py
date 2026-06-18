import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_partial_lse_kernel(
    A_ptr, B_ptr, bias_ptr,
    Mx_ptr, Sx_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_mxm, stride_mxn,
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
    sx_ptrs = Sx_ptr + offs_m * stride_mxm + pid_n * stride_mxn
    tl.store(mx_ptrs, row_max, mask=mask_m)
    tl.store(sx_ptrs, row_sum, mask=mask_m)


@triton.jit
def lse_reduce_act_kernel(
    Mx_ptr, Sx_ptr, out_ptr,
    M, NB,
    stride_mxm,
    BLOCK_NB: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs = tl.arange(0, BLOCK_NB)
    mask = offs < NB
    neg_inf = float('-inf')
    mx = tl.load(Mx_ptr + pid * stride_mxm + offs, mask=mask, other=neg_inf)
    sx = tl.load(Sx_ptr + pid * stride_mxm + offs, mask=mask, other=0.0)

    gmax = tl.max(mx, axis=0)
    safe_gmax = tl.where(gmax == neg_inf, 0.0, gmax)
    adj = sx * tl.exp(mx - safe_gmax)
    adj = tl.where(mask, adj, 0.0)
    total = tl.sum(adj, axis=0)
    lse = tl.log(total) + safe_gmax

    # Two LeakyReLU(0.01)
    x = lse
    x = tl.where(x >= 0.0, x, x * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

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
        W = self.linear.weight
        b = self.linear.bias
        M, K = x.shape
        N = self.out_features

        A = x
        B = W
        stride_am, stride_ak = A.stride(0), A.stride(1)
        stride_bk = 1
        stride_bn = K

        bias_t = b if b is not None else torch.zeros(N, device=x.device, dtype=x.dtype)

        # We need NB known to size Mx/Sx; pick a fixed BLOCK_N partitioning matching autotune
        # by allocating max possible NB based on smallest BLOCK_N (64).
        # Better: pre-pick BLOCK_N candidates—use 128 partitioning for buffer, autotune chooses internal block.
        # We'll allocate per chosen BLOCK_N via meta. Use lambda grid.
        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # Allocate worst-case for smallest BLOCK_N in configs (64)
        MAX_NB = triton.cdiv(N, 64)
        Mx = torch.empty((M, MAX_NB), device=x.device, dtype=torch.float32)
        Sx = torch.empty((M, MAX_NB), device=x.device, dtype=torch.float32)

        gemm_partial_lse_kernel[grid](
            A, B, bias_t,
            Mx, Sx,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            Mx.stride(0), Mx.stride(1),
        )

        # Determine actual NB used by the autotuner's chosen BLOCK_N
        best_cfg = gemm_partial_lse_kernel.best_config
        BLOCK_N_chosen = best_cfg.kwargs['BLOCK_N']
        NB = triton.cdiv(N, BLOCK_N_chosen)

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_NB = _next_pow2(NB)
        lse_reduce_act_kernel[(M,)](
            Mx, Sx, out,
            M, NB,
            Mx.stride(0),
            BLOCK_NB=BLOCK_NB,
        )
        return out