import torch
import torch.nn as nn
import triton
import triton.language as tl


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
    # Each program computes a row tile's sum over N of (x @ w.T + b)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k
        mask_k = k_idx < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_idx[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_idx[None, :] * stride_wk
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        # x: (BM, BK), w: (BN, BK) -> need (BM, BN) = x @ w.T
        acc += tl.dot(x, tl.trans(w))

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = tl.where(mask_n[None, :], acc, 0.0)
    # row-sum over the N-tile
    row_sum = tl.sum(acc, axis=1)  # (BLOCK_M,)
    # atomic add into output[m]
    tl.atomic_add(out_ptr + offs_m, row_sum, mask=mask_m)


def fused_linear_rowsum(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.zeros(M, device=x.device, dtype=torch.float32)
    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    fused_linear_sum_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
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