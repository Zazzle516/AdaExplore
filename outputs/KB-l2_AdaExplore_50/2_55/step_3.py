import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_pool_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes a (BLOCK_M, BLOCK_N) tile of the linear output,
    # then reduces it along N (with maxpool of KERNEL_SIZE) and atomic-adds
    # to per-row scalar accumulator.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    # mask out-of-bounds N to -inf for max pool
    neg_inf = float('-inf')
    acc = tl.where(mask_n[None, :], acc, neg_inf)

    # Reshape acc to (BLOCK_M, BLOCK_N // KERNEL_SIZE, KERNEL_SIZE) and reduce max over last
    # Then sum over the middle dim. We do this manually since BLOCK_N is multiple of KERNEL_SIZE.
    # We'll implement by iterating
    POOLED: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc_r = tl.reshape(acc, (BLOCK_M, POOLED, KERNEL_SIZE))
    pooled = tl.max(acc_r, axis=2)  # (BLOCK_M, POOLED)

    # Replace any -inf (entirely masked windows) with 0 so they don't contribute
    pooled = tl.where(pooled == neg_inf, 0.0, pooled)

    # Sum over POOLED dim -> (BLOCK_M,)
    row_sum = tl.sum(pooled, axis=1) * SCALE

    # atomic add to out_ptr[offs_m]
    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        # match nn.Linear init
        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())

    def forward(self, x):
        x = x.contiguous().cuda()
        M = x.shape[0]
        K = x.shape[1]
        N = self.out_features
        # ensure N is divisible by kernel_size for pooling; if not, fall back
        assert N % self.kernel_size == 0, "out_features must be divisible by kernel_size"

        w = self.weight.contiguous()  # (N, K)
        b = self.bias.contiguous()  # (N,)

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        fused_linear_pool_sum_kernel[grid](
            x, w, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w.stride(0), w.stride(1),
            SCALE=self.scale_factor,
            KERNEL_SIZE=self.kernel_size,
        )
        return out