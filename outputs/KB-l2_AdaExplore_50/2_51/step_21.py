import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The key insight: after GEMM, subtract, mean over dim=1, logsumexp over dim=1 (which is size 1),
# the result is a scalar per batch row. logsumexp of a single element is just that element.
# So x becomes shape (B, 1) = mean of (gemm(x) - subtract) over out_features.
# Then GELU, then add to original (B, in_features) -> broadcasts.
#
# mean over dim=1 of (W @ x + b - s) = (1/N) * sum_j (sum_k W[j,k]*x[k] + b[j] - s[j])
#                                    = (1/N) * (sum_k x[k] * sum_j W[j,k] + sum_j(b[j]-s[j]))
# But safety contract says no graph-level shortcuts that fold reductions into weights.
# So we must actually do the GEMM at runtime.
#
# Strategy: do a fused kernel that computes per-row sum of (W @ x + b - s),
# i.e. for each row i, compute sum_j (sum_k W[j,k] * x[i,k]) + sum_j(b[j]-s[j]).
# We compute the full GEMM tile by tile and reduce along j.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, bs_ptr,
    rowsum_ptr,             # (M,) atomic-accumulated row sums
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # GROUP_M swizzle for L2 reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs + k * stride_xk)
        w = tl.load(w_ptrs + k * stride_wk)
        acc += tl.dot(x, tl.trans(w))

    bs = tl.load(bs_ptr + offs_n)
    acc = acc + bs[None, :]
    row_partial = tl.sum(acc, axis=1)  # (BLOCK_M,)

    # atomic add into the (M,) row sum buffer
    tl.atomic_add(rowsum_ptr + offs_m, row_partial)


@triton.jit
def finalize_kernel(
    rowsum_ptr,   # (M,)
    orig_ptr,     # (M, K)
    out_ptr,      # (M, K)
    M, K, N,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    total = tl.load(rowsum_ptr + pid_m)
    mean_val = total / N
    inv_sqrt2 = 0.70710678118654752440
    gelu_val = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val * inv_sqrt2))

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    orig = tl.load(orig_ptr + pid_m * K + offs_k, mask=mask_k, other=0.0)
    out = orig + gelu_val
    tl.store(out_ptr + pid_m * K + offs_k, out, mask=mask_k)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.contiguous().cuda()
        original_x = x.clone()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        if self.gemm.bias is not None:
            bs = self.gemm.bias - self.subtract
        else:
            bs = -self.subtract
        bs = bs.contiguous()

        # rowsum buffer initialized to zero (atomic add target)
        rowsum = torch.zeros((M,), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        gemm_rowsum_kernel[grid](
            x, W, bs, rowsum,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        out = torch.empty_like(original_x)
        BLOCK_K_FIN = 1024
        grid_fin = (M, triton.cdiv(K, BLOCK_K_FIN))
        finalize_kernel[grid_fin](
            rowsum, original_x, out,
            M, K, N,
            BLOCK_K=BLOCK_K_FIN,
            num_warps=8,
        )
        return out