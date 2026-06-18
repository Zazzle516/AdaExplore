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
    # One program per M tile; reduces over N to produce out shape (M,)
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Accumulator: per-row sum of linear output
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)

    # Iterate over N tiles
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Compute GEMM tile: x[M_tile, K] @ w[N_tile, K].T -> [M_tile, N_tile]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            offs_k_cur = k_start + offs_k
            mask_k = offs_k_cur < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k_cur[None, :] * stride_xk
            x_mask = mask_m[:, None] & mask_k[None, :]
            x_block = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k_cur[None, :] * stride_wk
            w_mask = mask_n[:, None] & mask_k[None, :]
            w_block = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x_block, tl.trans(w_block))

        # add bias
        b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        acc += b_vals[None, :]

        # mask invalid N entries to 0 before summing
        acc = tl.where(mask_n[None, :], acc, 0.0)
        row_sum += tl.sum(acc, axis=1)

    tl.store(out_ptr + offs_m, row_sum, mask=mask_m)


def fused_linear_sum(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M),)
    fused_linear_sum_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
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
        w = self.linear.weight.contiguous().cuda()
        b = self.linear.bias.contiguous().cuda()

        # Fused linear + sum over out_features -> (batch,)
        s = fused_linear_sum(x, w, b)  # (batch_size,)
        # After sum (B,1), max over dim=1 keepdim -> (B,1) same values
        # mean over dim=1 keepdim -> (B,1) same values
        # logsumexp over dim=1 keepdim with single elem -> same values
        # so result is just s reshaped to (B,1)
        return s.view(-1, 1)