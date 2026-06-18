import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowmax_splitk_kernel(
    X_ptr, W_ptr, B_ptr, PART_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        mask_k = offs_k < k_remain
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = tl.where(mask_n[None, :], acc, -float('inf'))
    row_max = tl.max(acc, axis=1)

    # partials shape: (num_n_tiles, M); row-major, stride M
    num_m = tl.num_programs(0)
    out_ptrs = PART_ptr + pid_n * M + offs_m
    tl.store(out_ptrs, row_max, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.gemm.weight.contiguous().cuda()
        B = self.gemm.bias.contiguous().cuda()

        M = x.shape[0]
        K = self.in_features
        N = self.out_features

        if self.max_dim == 1:
            # Allocate partials big enough for the smallest BLOCK_N in the autotune set (64).
            num_n_upper = triton.cdiv(N, 64)
            partials = torch.full((num_n_upper, M), float('-inf'), device=x.device, dtype=torch.float32)

            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
            gemm_rowmax_splitk_kernel[grid](
                x, W, B, partials,
                M, N, K,
                x.stride(0), x.stride(1),
                W.stride(0), W.stride(1),
            )
            # Reduce partials across N-tile dimension to get the true rowmax.
            row_max = partials.max(dim=0).values.unsqueeze(1)  # (M, 1)
            # Subtract per-row mean over dim=1 (which is rowmax itself for a single column).
            y = row_max - row_max.mean(dim=1, keepdim=True)
            return torch.nn.functional.gelu(y)
        else:
            y = torch.nn.functional.linear(x, self.gemm.weight, self.gemm.bias)
            y = torch.max(y, dim=self.max_dim, keepdim=True).values
            y = y - y.mean(dim=1, keepdim=True)
            return torch.nn.functional.gelu(y)