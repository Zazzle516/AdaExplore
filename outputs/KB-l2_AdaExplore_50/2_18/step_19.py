import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)

    m_mask = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        cur_n = n_start + offs_n
        n_mask = cur_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            cur_k = k_start + offs_k
            k_mask = cur_k < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + cur_k[None, :] * stride_xk
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

            w_ptrs = w_ptr + cur_k[:, None] * stride_wk + cur_n[None, :] * stride_wn
            w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

            acc += tl.dot(x_tile, w_tile)

        b_vals = tl.load(b_ptr + cur_n, mask=n_mask, other=0.0)
        acc += b_vals[None, :]
        acc = tl.where(n_mask[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    tl.store(out_ptr + offs_m, row_acc, mask=m_mask)


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

        w = self.linear.weight  # (N, K)
        b = self.linear.bias    # (N,)
        # We want W as (K, N) contiguous for tl.dot
        wT = w.t().contiguous()

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        linear_sum_kernel[grid](
            x, wT, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            wT.stride(0), wT.stride(1),
        )
        # sum -> max(dim=1) on (M,1) is identity, mean(dim=1) on (M,1) identity,
        # logsumexp on (M,1) is identity (log(exp(v)) = v).
        return out.unsqueeze(1)