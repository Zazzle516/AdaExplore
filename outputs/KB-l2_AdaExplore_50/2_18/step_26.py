import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    x_ptr, w_ptr, b_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    SPLIT_N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (num_pid_m, SPLIT_N)
    pid_m = tl.program_id(0)
    pid_split = tl.program_id(1)

    # Each split handles a contiguous chunk of N
    n_per_split = N // SPLIT_N
    n_start = pid_split * n_per_split
    n_end = n_start + n_per_split

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    m_mask = offs_m < M

    # row accumulator across N
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over N tiles within this split
    for n_tile_start in range(n_start, n_end, BLOCK_N):
        offs_n = n_tile_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_mask = (offs_k + k) < K
            x_vals = tl.load(x_ptrs + k * stride_xk,
                             mask=m_mask[:, None] & k_mask[None, :], other=0.0)
            w_vals = tl.load(w_ptrs + k * stride_wk,
                             mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            acc += tl.dot(x_vals, tl.trans(w_vals))

        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += b_vals[None, :]
        acc = tl.where(n_mask[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    # atomic add into out[m]
    tl.atomic_add(out_ptr + offs_m, row_acc, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features

        W = self.linear.weight.contiguous()
        B = self.linear.bias.contiguous()

        out = torch.zeros((M,), device=x.device, dtype=torch.float32)

        SPLIT_N = 4  # split N across programs to increase parallelism

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), SPLIT_N)

        fused_gemm_rowsum_kernel[grid](
            x, W, B,
            out,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            SPLIT_N=SPLIT_N,
        )

        return out.view(M, 1)