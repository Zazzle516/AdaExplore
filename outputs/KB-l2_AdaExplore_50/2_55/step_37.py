import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
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
    # Each program owns BLOCK_M rows and loops over the full N dimension.
    # Within the loop, it computes BLOCK_M x BLOCK_N tile of X @ W.T + B,
    # max-pools along N with KERNEL_SIZE, sums the pooled values, and accumulates
    # the per-row pooled-sum in registers. Final result is scaled and stored.
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    pooled_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    POOLED: tl.constexpr = BLOCK_N // KERNEL_SIZE

    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + offs_n_base

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # add bias
        b = tl.load(B_ptr + offs_n)
        acc = acc + b[None, :]

        # max pool along N with kernel size KERNEL_SIZE.
        acc_r = tl.reshape(acc, (BLOCK_M, POOLED, KERNEL_SIZE))
        pooled = tl.max(acc_r, axis=2)  # (BLOCK_M, POOLED)

        # sum along POOLED axis and accumulate
        pooled_sum += tl.sum(pooled, axis=1)

    pooled_sum = pooled_sum * SCALE
    tl.store(Out_ptr + offs_m, pooled_sum, mask=mask_m)


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

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
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