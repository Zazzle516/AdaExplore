import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_pool_sum_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SCALE: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program computes a BLOCK_M x BLOCK_N tile of X @ W.T + B
    # Then performs max-pool over N (with kernel KERNEL_SIZE), sums result, scales,
    # and atomically adds partial sums to Out_ptr[m].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k0 in range(0, K, BLOCK_K):
        k_rem = K - k0
        mask_k = offs_k < k_rem
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # mask invalid N positions to -inf for max-pool
    neg_inf = float('-inf')
    acc = tl.where(mask_n[None, :], acc, neg_inf)

    # max pool along N with kernel size KERNEL_SIZE. BLOCK_N must be multiple of KERNEL_SIZE
    # reshape to (BLOCK_M, BLOCK_N // KERNEL_SIZE, KERNEL_SIZE)
    POOLED: tl.constexpr = BLOCK_N // KERNEL_SIZE
    acc_r = tl.reshape(acc, (BLOCK_M, POOLED, KERNEL_SIZE))
    pooled = tl.max(acc_r, axis=2)  # (BLOCK_M, POOLED)

    # sum along POOLED axis
    s = tl.sum(pooled, axis=1)  # (BLOCK_M,)
    s = s * SCALE

    # atomically add to Out_ptr[offs_m]
    tl.atomic_add(Out_ptr + offs_m, s, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous().cuda()
        B = self.matmul.bias.contiguous().cuda()

        M, K = x.shape
        N = self.out_features
        assert N % self.kernel_size == 0, "out_features must be divisible by kernel_size"

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        fused_linear_pool_sum_kernel[grid](
            x, W, B, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SCALE=self.scale_factor,
            KERNEL_SIZE=self.kernel_size,
        )

        return out