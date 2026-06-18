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
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            mask_k = k_idx < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            x_mask = mask_m[:, None] & mask_k[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_ptr + k_idx[:, None] * stride_wk + offs_n[None, :] * stride_wn
            w_mask = mask_k[:, None] & mask_n[None, :]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile)

        b_tile = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + b_tile[None, :]
        acc = tl.where(mask_n[None, :], acc, 0.0)

        row_acc += tl.sum(acc, axis=1)

    tl.store(out_ptr + offs_m, row_acc, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        # W is (out_features, in_features); we need it as (K, N) for tl.dot
        W = self.linear.weight  # (N, K)
        Wt = W.t().contiguous()  # (K, N)
        b = self.linear.bias.contiguous()

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_linear_sum_kernel[grid](
            x, Wt, b, out,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
        )

        # The remaining ops (max, mean, logsumexp, logsumexp) on dim=1 of (M,1)
        # are all identity on a length-1 dim except logsumexp which is also identity:
        # logsumexp over single element = that element.
        return out.unsqueeze(1)