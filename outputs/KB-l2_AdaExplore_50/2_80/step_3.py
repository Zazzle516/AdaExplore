import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile over (M, N), then we do partial max over N tile and atomically max into out_ptr[m]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x_mask = mask_m[:, None] & (offs_k[None, :] < k_remaining)
        w_mask = mask_n[:, None] & (offs_k[None, :] < k_remaining)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # add bias
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # mask invalid lanes to -inf for max
    acc = tl.where(mask_n[None, :], acc, -float('inf'))

    # row max within tile
    row_max = tl.max(acc, axis=1)  # [BLOCK_M]

    # atomic max into out
    tl.atomic_max(out_ptr + offs_m, row_max, mask=mask_m)


@triton.jit
def gelu_kernel(
    inp_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    # After max-keepdim along dim=1 yields shape (B,1), then x - mean over dim=1 of a (B,1) tensor is 0.
    # So input to gelu is 0 -> output is 0. But we still compute properly for correctness.
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    tl.store(out_ptr + offs, y, mask=mask)


def gemm_rowmax(x, weight, bias):
    M, K = x.shape
    N, Kw = weight.shape
    assert K == Kw
    # Initialize output to -inf for atomic max
    out = torch.full((M,), float('-inf'), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_rowmax_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, max_dim):
        super().__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.gemm.weight.contiguous().cuda()
        bias = self.gemm.bias.contiguous().cuda()

        if self.max_dim == 1:
            # Output of max over dim=1 keepdim=True -> shape (B, 1)
            # Then x - x.mean(dim=1, keepdim=True): mean of (B,1) along dim=1 = itself, so result is zeros
            # gelu(0) = 0. So output is zeros of shape (B, 1).
            # But we must execute the ops at runtime per safety contract.
            row_max = gemm_rowmax(x, weight, bias)  # (B,)
            row_max = row_max.unsqueeze(1)  # (B, 1)
            mean = row_max.mean(dim=1, keepdim=True)
            diff = row_max - mean
            # apply gelu via triton kernel
            out = torch.empty_like(diff)
            n = diff.numel()
            BLOCK = 256
            grid = (triton.cdiv(n, BLOCK),)
            gelu_kernel[grid](diff, out, n, BLOCK_SIZE=BLOCK)
            return out
        else:
            # max over dim=0 keepdim=True -> shape (1, N)
            y = torch.addmm(bias, x, weight.t())
            y = torch.max(y, dim=0, keepdim=True).values
            y = y - y.mean(dim=1, keepdim=True)
            out = torch.empty_like(y)
            n = y.numel()
            BLOCK = 256
            grid = (triton.cdiv(n, BLOCK),)
            gelu_kernel[grid](y, out, n, BLOCK_SIZE=BLOCK)
            return out