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
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        # x: (BM, BK), w: (BN, BK) -> need x @ w.T = (BM, BN)
        acc += tl.dot(x, tl.trans(w), input_precision='ieee')
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias for this N tile
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b[None, :]

    # mask out invalid N
    acc = tl.where(mask_n[None, :], acc, 0.0)

    # row-sum over BLOCK_N
    row_sum = tl.sum(acc, axis=1)  # (BLOCK_M,)

    # atomic add into Out[offs_m]
    tl.atomic_add(Out_ptr + offs_m, row_sum, mask=mask_m)


def fused_linear_rowsum(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.zeros(M, device=x.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
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
        # Fused linear + row-sum -> (batch, 1)
        s = fused_linear_rowsum(x, w, b)  # (batch, 1)
        # max over dim=1 keepdim -> same shape (since size 1)
        s = torch.max(s, dim=1, keepdim=True)[0]
        # mean over dim=1 keepdim
        s = torch.mean(s, dim=1, keepdim=True)
        # logsumexp twice over dim=1 size 1 = identity (log(exp(x)) = x)
        s = torch.logsumexp(s, dim=1, keepdim=True)
        s = torch.logsumexp(s, dim=1, keepdim=True)
        return s