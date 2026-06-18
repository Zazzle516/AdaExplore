import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The forward computes:
# x = gemm(x) -> (B, out_features)
# x = x - subtract
# x = mean(x, dim=1, keepdim=True) -> (B, 1)
# x = logsumexp(x, dim=1, keepdim=True) -> (B, 1) (since dim=1 size is 1, this is just x)
# x = gelu(x) -> (B, 1)
# x = x + original_x -> (B, in_features) via broadcasting
#
# So effectively we need s[b] = gelu(mean_over_j(W[j,:] @ x[b,:] + bias[j] - sub[j]))
# and output[b, i] = s[b] + x[b, i]
#
# mean_j (W[j,:] @ x[b,:] + bias[j] - sub[j])
# = (1/out_features) * sum_j sum_k W[j,k]*x[b,k] + (1/out_features)*sum_j(bias[j]-sub[j])
# But per safety contract, we should NOT precompute reduced weights. We must
# execute the full GEMM at runtime.
#
# Strategy: 
# 1. Compute GEMM tile producing (B, out_features), but fuse subtraction and 
#    per-row reduction into a single kernel that outputs s[b] (B,) instead of materializing full (B, out_features).
# 2. Then a second kernel computes output[b, i] = gelu(s[b]) + x[b, i]
#
# For step 1: each program handles a (BLOCK_M) rows tile. It accumulates sum over j of
# (W[j,k] @ x[b,k] + bias[j] - sub[j]) but we still do the full GEMM operation - 
# we just reduce instead of storing intermediate.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_subtract_mean_kernel(
    x_ptr, w_ptr, b_ptr, sub_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per (M tile). It loops over N tiles, computing full GEMM tile via tl.dot,
    # adding bias - sub, and accumulating row-sums. Final result: out[m] = sum_n / N.
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    # Per-row accumulator (reduction over N)
    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Loop over N tiles
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        # Compute tile of GEMM: (BLOCK_M, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k_start in range(0, K, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            # Load x: (BLOCK_M, BLOCK_K)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
            x_mask = mask_m[:, None] & mask_k[None, :]
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load w: (BLOCK_K, BLOCK_N) - W is (N, K), we want W[n,k] transposed
            w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
            w_mask = mask_k[:, None] & mask_n[None, :]
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += tl.dot(x_tile, w_tile)

        # Add bias and subtract
        bias_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
        sub_vals = tl.load(sub_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias_vals[None, :] - sub_vals[None, :]

        # Mask out-of-bounds N
        acc = tl.where(mask_n[None, :], acc, 0.0)

        # Sum over N tile
        row_sum += tl.sum(acc, axis=1)

    # Divide by N for mean
    mean_val = row_sum / N

    # Apply GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * mean_val * (1.0 + tl.erf(mean_val * inv_sqrt2))

    # Store
    tl.store(out_ptr + offs_m, gelu_val, mask=mask_m)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def add_scalar_kernel(
    x_ptr, s_ptr, out_ptr,
    M, K,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    s_val = tl.load(s_ptr + pid_m)
    offs_k = pid_k * BLOCK + tl.arange(0, BLOCK)
    mask_k = offs_k < K
    x_vals = tl.load(x_ptr + pid_m * K + offs_k, mask=mask_k, other=0.0)
    out_vals = x_vals + s_val
    tl.store(out_ptr + pid_m * K + offs_k, out_vals, mask=mask_k)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        K = x.shape[1]
        N = self.out_features

        W = self.gemm.weight.contiguous()  # (N, K)
        bias = self.gemm.bias.contiguous() if self.gemm.bias is not None else torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        s = torch.empty((B,), device=x.device, dtype=torch.float32)

        grid1 = lambda meta: (triton.cdiv(B, meta['BLOCK_M']),)
        gemm_subtract_mean_kernel[grid1](
            x, W, bias, sub, s,
            B, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        out = torch.empty_like(x)
        grid2 = lambda meta: (B, triton.cdiv(K, meta['BLOCK']))
        add_scalar_kernel[grid2](x, s, out, B, K)

        return out