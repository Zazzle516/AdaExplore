import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _gemm_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M

    # Per-row accumulator (sum over all N of bias-added GEMM output)
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # bias pointer offsets
    n_range = tl.arange(0, BLOCK_N)

    num_n = tl.cdiv(N, BLOCK_N)
    for n_idx in range(0, num_n):
        offs_n = n_idx * BLOCK_N + n_range
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

        for k in range(0, K, BLOCK_K):
            k_remaining = K - k
            mask_k = offs_k < k_remaining
            x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc += tl.dot(x, tl.trans(w), input_precision='tf32')
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
        acc += b[None, :]
        acc = tl.where(mask_n[None, :], acc, 0.0)

        row_acc += tl.sum(acc, axis=1)

    tl.store(Out_ptr + offs_m, row_acc, mask=mask_m)


def fused_linear_rowsum(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty(M, device=x.device, dtype=torch.float32)

    BLOCK_M = 64
    BLOCK_N = 128
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M),)
    _gemm_rowsum_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out.unsqueeze(1)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous().cuda()
        b = self.linear.bias.contiguous().cuda()
        s = fused_linear_rowsum(x, w, b)
        s = torch.max(s, dim=1, keepdim=True)[0]
        s = torch.mean(s, dim=1, keepdim=True)
        s = torch.logsumexp(s, dim=1, keepdim=True)
        s = torch.logsumexp(s, dim=1, keepdim=True)
        return s