import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_online_lse_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    One program per row-tile (BLOCK_M rows). Streams all N columns in BLOCK_N
    chunks, performing a full GEMM tile then folding into an online LSE
    accumulator (running max + running sumexp). After streaming all of N,
    apply the LeakyReLU/LeakyReLU/GELU/GELU fusion and store one scalar per row.
    """
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    # Online LSE state
    run_max = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    run_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    num_n_blocks = tl.cdiv(N, BLOCK_N)

    for nb in range(0, num_n_blocks):
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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

        # Block max
        blk_max = tl.max(acc, axis=1)  # [BLOCK_M]
        new_max = tl.maximum(run_max, blk_max)
        safe_new_max = tl.where(new_max == neg_inf, 0.0, new_max)

        # Rescale old sum
        scale = tl.exp(tl.where(run_max == neg_inf, 0.0, run_max) - safe_new_max)
        # Mask invalid entries to 0 before exp
        exp_vals = tl.exp(acc - safe_new_max[:, None])
        exp_vals = tl.where(mask_m[:, None] & mask_n[None, :], exp_vals, 0.0)
        blk_sum = tl.sum(exp_vals, axis=1)

        run_sum = run_sum * scale + blk_sum
        run_max = new_max

    safe_max = tl.where(run_max == float('-inf'), 0.0, run_max)
    lse = tl.log(run_sum) + safe_max

    # Fused activations: LeakyReLU(0.01) x2 then GELU x2
    x = lse
    x = tl.where(x >= 0.0, x, x * 0.0001)  # two leakyrelu collapsed

    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs_m, x, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight  # [N, K]
        b = self.linear.bias
        M, K = x.shape
        N = self.out_features

        if b is None:
            b = torch.zeros(N, device=x.device, dtype=x.dtype)

        # A: [M, K] row-major
        stride_am, stride_ak = x.stride(0), x.stride(1)
        # B = W viewed as [K, N]: W[n, k] => stride_bk=1, stride_bn=K
        stride_bk = 1
        stride_bn = K

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

        gemm_online_lse_kernel[grid](
            x, W, b, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
        )
        return out