import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


# Split-K GEMM kernel: computes partial sums of x @ W^T
# Each program handles a (M_tile, N_tile) for a slice of K
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def gemm_splitk_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on, stride_ok,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # K range for this split
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + (k_start + offs_k)[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    num_iters = tl.cdiv(k_end - k_start, BLOCK_K)
    for k in range(0, num_iters):
        k_offs = k_start + k * BLOCK_K + offs_k
        mask_k = k_offs < k_end
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on + pid_k * stride_ok
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# Reduce-K + bias + scale*2 + clamp + LSE + mish fusion
# Each program handles one row, reading partial sums along K split and reducing to lse, then mish.
@triton.jit
def reduce_lse_mish_kernel(
    partial_ptr,   # [SPLIT_K, M, N]
    bias_ptr,      # [N]
    out_ptr,       # [M, 1]
    M, N,
    stride_pk, stride_pm, stride_pn,
    SPLIT_K: tl.constexpr,
    SCALE2: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)

    # First pass: compute max over N
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs_n
        mask = idx < N

        # accumulate over split-k
        v = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in range(0, SPLIT_K):
            p = tl.load(
                partial_ptr + k * stride_pk + pid_m * stride_pm + idx * stride_pn,
                mask=mask, other=0.0
            )
            v += p
        # add bias
        b = tl.load(bias_ptr + idx, mask=mask, other=0.0)
        v = v + b
        # scale * 2
        v = v * SCALE2
        # clamp
        v = tl.minimum(tl.maximum(v, CMIN), CMAX)
        v = tl.where(mask, v, -float('inf'))

        cur_max = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, cur_max)

    # Second pass: sum exp
    sum_exp = tl.full((), 0.0, dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        idx = n_start + offs_n
        mask = idx < N

        v = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in range(0, SPLIT_K):
            p = tl.load(
                partial_ptr + k * stride_pk + pid_m * stride_pm + idx * stride_pn,
                mask=mask, other=0.0
            )
            v += p
        b = tl.load(bias_ptr + idx, mask=mask, other=0.0)
        v = v + b
        v = v * SCALE2
        v = tl.minimum(tl.maximum(v, CMIN), CMAX)

        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_exp += tl.sum(e, axis=0)

    lse = max_val + tl.log(sum_exp)

    # mish: lse * (lse * tanh(softplus(lse)))
    sp = tl.where(lse > 0, lse, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(lse)))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    result = lse * lse * tanh_sp

    tl.store(out_ptr + pid_m, result)


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super().__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = float(scale_factor)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.SPLIT_K = 4

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous().cuda()
        b = self.matmul.bias.contiguous().cuda()

        M, K = x.shape
        N = W.shape[0]
        SPLIT_K = self.SPLIT_K

        partial = torch.empty((SPLIT_K, M, N), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
            SPLIT_K,
        )
        gemm_splitk_kernel[grid](
            x, W, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(1), partial.stride(2), partial.stride(0),
            SPLIT_K=SPLIT_K,
        )

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
        BLOCK_N = 1024
        reduce_lse_mish_kernel[(M,)](
            partial, b, out,
            M, N,
            partial.stride(0), partial.stride(1), partial.stride(2),
            SPLIT_K=SPLIT_K,
            SCALE2=self.scale_factor * 2.0,
            CMIN=self.clamp_min,
            CMAX=self.clamp_max,
            BLOCK_N=BLOCK_N,
            num_warps=8,
        )

        return out