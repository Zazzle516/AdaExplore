import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_lse_act_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    # Online LSE accumulators per row
    neg_inf = float('-inf')
    running_max = tl.full((BLOCK_M,), neg_inf, dtype=tl.float32)
    running_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    NB = tl.cdiv(N, BLOCK_N)
    KB = tl.cdiv(K, BLOCK_K)

    for nb in range(0, NB):
        offs_n = nb * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, KB):
            k_remaining = K - k * BLOCK_K
            mask_k = offs_k < k_remaining
            a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk

        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]
        acc = tl.where(mask_n[None, :], acc, neg_inf)

        block_max = tl.max(acc, axis=1)  # [BLOCK_M]
        new_max = tl.maximum(running_max, block_max)
        safe_new_max = tl.where(new_max == neg_inf, 0.0, new_max)
        # exp(acc - new_max)
        exp_vals = tl.exp(acc - safe_new_max[:, None])
        exp_vals = tl.where(mask_n[None, :], exp_vals, 0.0)
        block_sum = tl.sum(exp_vals, axis=1)
        # scale running sum
        scale = tl.exp(running_max - safe_new_max)
        scale = tl.where(running_max == neg_inf, 0.0, scale)
        running_sum = running_sum * scale + block_sum
        running_max = new_max

    safe_max = tl.where(running_max == neg_inf, 0.0, running_max)
    lse = tl.log(running_sum) + safe_max

    # Two LeakyReLU(0.01) collapsed to slope 1e-4 for negatives
    x = tl.where(lse >= 0.0, lse, lse * 1e-4)

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

        A = x
        B = W
        stride_am, stride_ak = A.stride(0), A.stride(1)
        # B as [K, N] via W [N, K]: B[k,n] = W[n,k]
        stride_bk = 1
        stride_bn = K

        bias_t = b if b is not None else torch.zeros(N, device=x.device, dtype=x.dtype)

        out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)

        fused_gemm_lse_act_kernel[grid](
            A, B, bias_t, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
        )
        return out