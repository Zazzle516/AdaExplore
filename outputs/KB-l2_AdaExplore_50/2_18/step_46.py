import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def colsum_kernel(
    w_ptr, b_ptr, out_ptr,
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
):
    # One program per K column index; reduces N rows of weight.
    pid_k = tl.program_id(0)
    if pid_k >= K:
        return

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # bias sum accumulator - only compute once (pid_k == 0)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        w_ptrs = w_ptr + offs_n * stride_wn + pid_k * stride_wk
        w = tl.load(w_ptrs, mask=mask_n, other=0.0)
        acc += w
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_k, s)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def biassum_kernel(
    b_ptr, out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += b
    s = tl.sum(acc, axis=0)
    tl.store(out_ptr, s)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 512}, num_warps=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 2048}, num_warps=8),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 1024}, num_warps=8),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 2048}, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 1024}, num_warps=8),
    ],
    key=['M', 'K'],
)
@triton.jit
def matvec_kernel(
    x_ptr, v_ptr, bs_ptr, out_ptr,
    M, K,
    stride_xm, stride_xk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        v = tl.load(v_ptr + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(x * v[None, :], axis=1)

    bs = tl.load(bs_ptr)
    acc = acc + bs
    tl.store(out_ptr + offs_m, acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous()  # (N, K)
        b = self.linear.bias.contiguous()    # (N,)
        M, K = x.shape
        N = w.shape[0]

        # Stage 1a: w_colsum[k] = sum_n w[n, k]  -> shape (K,)
        w_colsum = torch.empty(K, device=x.device, dtype=torch.float32)
        colsum_kernel[(K,)](
            w, b, w_colsum,
            N, K,
            w.stride(0), w.stride(1),
        )

        # Stage 1b: bias_sum = sum_n b[n] -> scalar
        bias_sum = torch.empty(1, device=x.device, dtype=torch.float32)
        biassum_kernel[(1,)](b, bias_sum, N)

        # Stage 2: out[m] = sum_k x[m,k] * w_colsum[k] + bias_sum
        out = torch.empty(M, device=x.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        matvec_kernel[grid](
            x, w_colsum, bias_sum, out,
            M, K,
            x.stride(0), x.stride(1),
        )

        return out.unsqueeze(1)