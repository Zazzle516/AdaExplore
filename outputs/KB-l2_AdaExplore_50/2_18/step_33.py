import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator: per-row sum of (x @ W^T)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    m_mask = offs_m < M

    # Loop over K in chunks
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K

        # Load x tile (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

        # Compute w_colsum (BLOCK_K,) by reducing W across N in tiles
        w_colsum = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for n_start in range(0, N, BLOCK_N):
            n_offs = n_start + offs_n
            n_mask = n_offs < N
            w_ptrs = w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
            w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
            w_colsum += tl.sum(w_tile, axis=0)

        # acc[m] += sum_k x_tile[m, k] * w_colsum[k]
        acc += tl.sum(x_tile * w_colsum[None, :], axis=1)

    # Add bias.sum() in epilogue
    b_offs = tl.arange(0, BLOCK_N)
    b_sum = tl.zeros((), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + b_offs
        n_mask = n_offs < N
        b_tile = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
        b_sum += tl.sum(b_tile, axis=0)

    acc += b_sum

    tl.store(out_ptr + offs_m, acc, mask=m_mask)


@triton.jit
def fused_kernel_v2(
    x_ptr, w_colsum_ptr, b_sum_ptr, out_ptr,
    M, K,
    stride_xm, stride_xk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        k_mask = k_offs < K
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        wc = tl.load(w_colsum_ptr + k_offs, mask=k_mask, other=0.0)
        acc += tl.sum(x_tile * wc[None, :], axis=1)

    bs = tl.load(b_sum_ptr)
    acc += bs
    tl.store(out_ptr + offs_m, acc, mask=m_mask)


@triton.jit
def w_colsum_kernel(
    w_ptr, out_ptr,
    N, K,
    stride_wn, stride_wk,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_k = tl.program_id(0)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = offs_k < K
    offs_n = tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs_n
        n_mask = n_offs < N
        w_ptrs = w_ptr + n_offs[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & k_mask[None, :], other=0.0)
        acc += tl.sum(w_tile, axis=0)

    tl.store(out_ptr + offs_k, acc, mask=k_mask)


@triton.jit
def bias_sum_kernel(
    b_ptr, out_ptr,
    N,
    BLOCK_N: tl.constexpr,
):
    offs = tl.arange(0, BLOCK_N)
    acc = tl.zeros((), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        n_offs = n_start + offs
        n_mask = n_offs < N
        b_tile = tl.load(b_ptr + n_offs, mask=n_mask, other=0.0)
        acc += tl.sum(b_tile, axis=0)
    tl.store(out_ptr, acc)


def fused_forward(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    # Compute w_colsum at runtime (reduces W across N for each K)
    w_colsum = torch.empty((K,), device=x.device, dtype=torch.float32)
    BLOCK_K_W = 128
    BLOCK_N_W = 128
    grid_w = ((K + BLOCK_K_W - 1) // BLOCK_K_W,)
    w_colsum_kernel[grid_w](
        weight, w_colsum,
        N, K,
        weight.stride(0), weight.stride(1),
        BLOCK_K=BLOCK_K_W, BLOCK_N=BLOCK_N_W,
        num_warps=4, num_stages=3,
    )

    # Compute bias.sum at runtime
    b_sum = torch.empty((), device=x.device, dtype=torch.float32)
    BLOCK_N_B = triton.next_power_of_2(min(N, 8192))
    if BLOCK_N_B > 4096:
        BLOCK_N_B = 4096
    bias_sum_kernel[(1,)](
        bias, b_sum,
        N,
        BLOCK_N=BLOCK_N_B,
        num_warps=4,
    )

    # Main fused kernel: acc[m] = sum_k x[m,k] * w_colsum[k] + b_sum
    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK_M = 32
    BLOCK_K = 128
    grid = ((M + BLOCK_M - 1) // BLOCK_M,)
    fused_kernel_v2[grid](
        x, w_colsum, b_sum, out,
        M, K,
        x.stride(0), x.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
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
        return fused_forward(x, w, b)