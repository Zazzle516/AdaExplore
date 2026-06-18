import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _row_sum_gemm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # N range for this program
    n_per_split = tl.cdiv(N, SPLIT_N)
    n_start = pid_n * n_per_split
    n_end = tl.minimum(n_start + n_per_split, N)

    # Per-row accumulator for sum over N
    row_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Bias contribution for this split
    offs_n_full = n_start + tl.arange(0, BLOCK_N)
    # We'll accumulate bias as we iterate N tiles

    num_n_tiles = tl.cdiv(n_end - n_start, BLOCK_N)

    for nt in range(0, num_n_tiles):
        offs_n = n_start + nt * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < n_end

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            k_offs = k_start + offs_k
            k_mask = k_offs < K

            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
            x_mask = (offs_m[:, None] < M) & (k_mask[None, :])
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_mask = (n_mask[:, None]) & (k_mask[None, :])
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, tl.trans(w_tile), allow_tf32=False)

        # Add bias
        b_vals = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc += b_vals[None, :]

        # Mask out invalid n
        acc = tl.where(n_mask[None, :], acc, 0.0)

        # Sum across N tile
        row_acc += tl.sum(acc, axis=1)

    # Atomic add into output
    m_mask = offs_m < M
    tl.atomic_add(out_ptr + offs_m, row_acc, mask=m_mask)


def fused_linear_rowsum(x, W, b):
    M, K = x.shape
    N = W.shape[0]
    out = torch.zeros(M, device=x.device, dtype=torch.float32)

    BLOCK_M = 32
    BLOCK_N = 64
    BLOCK_K = 64
    SPLIT_N = 16

    grid = (triton.cdiv(M, BLOCK_M), SPLIT_N)
    _row_sum_gemm_kernel[grid](
        x, W, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        W.stride(0), W.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_N=SPLIT_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        row_sum = fused_linear_rowsum(x, W, b)  # (M,)
        # max/mean/logsumexp/logsumexp along dim=1 with size 1 are all identity
        out = row_sum.unsqueeze(1)
        return out