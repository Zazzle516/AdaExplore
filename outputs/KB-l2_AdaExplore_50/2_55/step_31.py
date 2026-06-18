import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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
        w = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # pairwise max pool (kernel_size=2) along N axis
    # reshape (BLOCK_M, BLOCK_N) -> (BLOCK_M, BLOCK_N/2, 2)
    acc_r = tl.reshape(acc, (BLOCK_M, BLOCK_N // 2, 2))
    pooled = tl.max(acc_r, axis=2)  # (BLOCK_M, BLOCK_N/2)

    # mask out invalid n positions for the pooled values
    # pooled position p corresponds to original positions 2p, 2p+1
    # valid if both 2p and 2p+1 are < N. Since N is even (32768), all valid if pid_n*BLOCK_N + 2p+1 < N
    offs_p = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
    # valid pooled position if 2*offs_p + 1 < N, i.e., offs_p < N//2
    mask_p = offs_p < (N // 2)
    pooled = tl.where(mask_p[None, :], pooled, 0.0)

    # sum across the pooled N dimension
    row_sum = tl.sum(pooled, axis=1)  # (BLOCK_M,)
    row_sum = row_sum * scale

    # atomic add to out[m]
    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        assert self.kernel_size == 2, "This fused kernel assumes kernel_size=2"
        x = x.contiguous().cuda()
        W = self.matmul.weight.contiguous().cuda()  # (N, K)
        b = self.matmul.bias.contiguous().cuda()    # (N,)
        M, K = x.shape
        N = W.shape[0]
        assert N % 2 == 0

        out = torch.zeros(M, device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        fused_gemm_pool_sum_kernel[grid](
            x, W, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            self.scale_factor,
        )

        return out