import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, out_ptr,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
        acc += tl.dot(x, tl.trans(w), allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    row_partial = tl.sum(acc, axis=1)  # (BLOCK_M,)

    out_ptrs = out_ptr + pid_n * M + offs_m
    tl.store(out_ptrs, row_partial, mask=offs_m < M)


@triton.jit
def reduce_n_kernel(
    partial_ptr, bias_sum_ptr, out_ptr,
    M, GRID_N,
    BLOCK_GN: tl.constexpr,
):
    m = tl.program_id(0)
    offs = tl.arange(0, BLOCK_GN)
    mask = offs < GRID_N
    ptrs = partial_ptr + offs * M + m
    vals = tl.load(ptrs, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)
    bs = tl.load(bias_sum_ptr)
    tl.store(out_ptr + m, s + bs)


def fused_linear_rowsum(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape

    BLOCK_M = 64
    BLOCK_N = 256
    BLOCK_K = 32

    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N

    partial = torch.empty((grid_n, M), device=x.device, dtype=torch.float32)

    gemm_rowsum_kernel[(grid_m, grid_n)](
        x, weight, partial,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )

    bias_sum = bias.sum().to(torch.float32).reshape(())
    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    BLOCK_GN = triton.next_power_of_2(grid_n)
    reduce_n_kernel[(M,)](
        partial, bias_sum, out,
        M, grid_n,
        BLOCK_GN=BLOCK_GN,
        num_warps=1,
    )
    return out.reshape(M, 1)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.linear.weight.contiguous()
        b = self.linear.bias.contiguous()
        out = fused_linear_rowsum(x, w, b)
        return out