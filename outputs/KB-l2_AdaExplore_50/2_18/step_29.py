import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pn, stride_pm,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        # x: (BM, BK), w: (BN, BK) -> need (BM, BN) = x @ w.T
        acc += tl.dot(x, tl.trans(w), allow_tf32=False)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # row-reduce over BLOCK_N
    row_partial = tl.sum(acc, axis=1)  # (BLOCK_M,)

    # store partial into partial[pid_n, m]
    out_ptrs = partial_ptr + pid_n * stride_pn + offs_m * stride_pm
    tl.store(out_ptrs, row_partial, mask=mask_m)


@triton.jit
def reduce_partial_kernel(
    partial_ptr, bias_sum_ptr, out_ptr,
    M, GN,
    stride_pn, stride_pm,
    BLOCK_GN: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs = tl.arange(0, BLOCK_GN)
    mask = offs < GN
    vals = tl.load(partial_ptr + offs * stride_pn + pid * stride_pm, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)
    b = tl.load(bias_sum_ptr)
    tl.store(out_ptr + pid, s + b)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.linear.weight.contiguous()  # (N, K)
        b = self.linear.bias.contiguous()    # (N,)

        M, K = x.shape
        N = W.shape[0]

        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        grid_m = (M + BLOCK_M - 1) // BLOCK_M
        grid_n = (N + BLOCK_N - 1) // BLOCK_N

        partial = torch.empty((grid_n, M), device=x.device, dtype=torch.float32)

        gemm_rowsum_kernel[(grid_m, grid_n)](
            x, W, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        bias_sum = b.sum().to(torch.float32).reshape(())

        out = torch.empty((M,), device=x.device, dtype=torch.float32)

        # next power of two >= grid_n
        BLOCK_GN = 1
        while BLOCK_GN < grid_n:
            BLOCK_GN *= 2

        reduce_partial_kernel[(M,)](
            partial, bias_sum, out,
            M, grid_n,
            partial.stride(0), partial.stride(1),
            BLOCK_GN=BLOCK_GN,
            num_warps=4,
        )

        # downstream: max over dim1 of (M,1) -> same; mean over dim1 -> same;
        # logsumexp over singleton dim -> same value (log(exp(x))=x). So output = out reshape.
        return out.reshape(M, 1)