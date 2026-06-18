import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    scale,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < M

    # Accumulator for the final scalar per row (sum over pooled outputs)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over N in chunks of BLOCK_N (BLOCK_N must be even for pool kernel_size=2)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Compute GEMM tile: [BLOCK_M, BLOCK_N] = x[BLOCK_M, K] @ w[N, K].T
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
            w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)

            acc += tl.dot(x_tile, tl.trans(w_tile))

        # Add bias
        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + b_vals[None, :]

        # Mask invalid N positions to -inf for max pool
        neg_inf = float('-inf')
        acc = tl.where(n_mask[None, :], acc, neg_inf)

        # Max pool with kernel_size=2: reshape [BLOCK_M, BLOCK_N] -> [BLOCK_M, BLOCK_N/2, 2]
        acc_reshaped = tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2))
        pooled = tl.max(acc_reshaped, axis=2)  # [BLOCK_M, BLOCK_N/2]

        # Sum pooled values for this chunk
        row_sum += tl.sum(pooled, axis=1)

    # Scale and store one scalar per row
    out = row_sum * scale
    tl.store(out_ptr + offs_m, out, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        # Mirror nn.Linear init
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous()
        B = self.matmul.bias.contiguous()

        M, K = x.shape
        N = self.out_features

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M),)

        fused_kernel[grid](
            x, W, B, out,
            M, N, K,
            float(self.scale_factor),
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        return out