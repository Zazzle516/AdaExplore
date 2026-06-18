import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _row_sum_linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program computes a partial row-sum for BLOCK_M rows over a chunk of N columns.
    # Result: partial sum atomically added to out[m].
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # accumulator for the GEMM tile (BLOCK_M, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_base = x_ptr + offs_m[:, None] * stride_xm
    w_base = w_ptr + offs_n[:, None] * stride_wn

    for k0 in range(0, K, BLOCK_K):
        k_idx = k0 + offs_k
        k_mask = k_idx < K
        x_ptrs = x_base + k_idx[None, :] * stride_xk
        x_vals = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

        w_ptrs = w_base + k_idx[None, :] * stride_wk
        w_vals = tl.load(w_ptrs, mask=mask_n[:, None] & k_mask[None, :], other=0.0)
        # We want x @ W^T, so dot(x_vals (M,K), w_vals.T (K,N))
        acc += tl.dot(x_vals, tl.trans(w_vals), out_dtype=tl.float32)

    # add bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_vals[None, :]

    # mask out-of-bounds n columns to 0 before summing
    acc = tl.where(mask_n[None, :], acc, 0.0)

    # sum along N -> (BLOCK_M,)
    row_partial = tl.sum(acc, axis=1)

    # atomic add to out[offs_m]
    tl.atomic_add(out_ptr + offs_m, row_partial, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        assert x.is_cuda, "Input must be CUDA"
        x = x.contiguous()
        M = x.shape[0]
        K = x.shape[1]
        N = self.out_features

        W = self.linear.weight.contiguous()  # (N, K)
        b = self.linear.bias.contiguous()    # (N,)

        # output: per-row sum of (x @ W^T + b), shape (M,)
        row_sum = torch.zeros(M, device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _row_sum_linear_kernel[grid](
            x, W, b, row_sum,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # After sum(dim=1, keepdim=True) -> (M, 1)
        # max(dim=1) over single elt -> same
        # mean(dim=1) over single elt -> same
        # logsumexp(dim=1) over single elt -> same value
        # logsumexp again -> same
        # So final result is just row_sum reshaped to (M, 1)
        out = row_sum.unsqueeze(1).to(x.dtype)
        return out