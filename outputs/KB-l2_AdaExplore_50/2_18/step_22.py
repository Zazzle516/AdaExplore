import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes the full row-sum for a tile of BLOCK_M rows.
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    offs_k = tl.arange(0, BLOCK_K)
    offs_bn = tl.arange(0, BLOCK_N)

    # Per-row scalar accumulator
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + offs_bn
        mask_n = offs_n < N
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            mask_k = k_idx < K
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
            w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w))
        b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + b[None, :]
        acc = tl.where(mask_n[None, :], acc, 0.0)
        row_acc += tl.sum(acc, axis=1)

    tl.store(out_ptr + offs_m, row_acc, mask=mask_m)


def fused_linear_rowsum(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, device=x.device, dtype=torch.float32)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    fused_linear_sum_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        # Compute row-sum of (x @ w.T + b): shape (M,)
        s = fused_linear_rowsum(x, w, b)  # (M,)
        # After sum: (M,1). max over dim=1 of (M,1) -> (M,1). mean -> (M,1).
        # logsumexp on a single element is just the element itself. Twice.
        return s.unsqueeze(1)