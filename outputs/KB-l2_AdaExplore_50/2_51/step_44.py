import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The key insight: after gemm -> subtract -> mean(dim=1) -> logsumexp(dim=1) -> gelu,
# we get a [batch_size, 1] tensor. Each element is a scalar function of one row.
# 
# For a row x of input:
#   y = gemm(x) - subtract   -> shape [out_features]
#   m = mean(y) = (sum(W @ x) + sum(b) - sum(subtract)) / out_features
#   lse(m) over dim=1 with size 1 = m  (logsumexp of single element is itself)
#   g = gelu(m)
#   out = g + original_x  (broadcast scalar to [in_features])
#
# But we must execute every op at runtime per safety contract. So we compute
# gemm fully, then do the reductions and elementwise ops with fused kernels.


# Fused GEMM + row-sum reduction kernel.
# Computes acc = X @ W^T + b - subtract for a [BLOCK_M, BLOCK_N] tile,
# then sums over the N axis and atomic-adds into RowSum[M].
# X: [M, K] contiguous (stride_xk=1)
# W: [N, K] contiguous (stride_wk=1) -- load as [BLOCK_K, BLOCK_N] directly
# b, sub: [N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    X_ptr, W_ptr, b_ptr, sub_ptr, RowSum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # X: [BLOCK_M, BLOCK_K]
    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # W: [N, K]; load tile as [BLOCK_K, BLOCK_N] -> ptrs[k, n] = W[offs_n[n], offs_k[k]]
    w_ptrs = W_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    s = tl.load(sub_ptr + offs_n, mask=offs_n < N, other=0.0)
    n_valid = offs_n < N
    acc = acc + (b - s)[None, :]
    # zero out invalid N lanes so the reduction is correct
    acc = tl.where(n_valid[None, :], acc, 0.0)

    # Reduce over N within this tile, then atomic-add into RowSum[offs_m]
    row_partial = tl.sum(acc, axis=1)  # [BLOCK_M]
    m_mask = offs_m < M
    tl.atomic_add(RowSum_ptr + offs_m, row_partial, mask=m_mask)


# Fused: mean = rowsum / N; gelu(mean); broadcast-add to original_x
@triton.jit
def gelu_residual_kernel(
    X_ptr, RowSum_ptr, OUT_ptr,
    M, K, N,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K
    rs = tl.load(RowSum_ptr + pid_m)
    mean = rs / N
    # logsumexp over single element = mean
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * mean * (1.0 + tl.math.erf(mean * inv_sqrt2))
    x = tl.load(X_ptr + pid_m * K + offs_k, mask=mask, other=0.0)
    tl.store(OUT_ptr + pid_m * K + offs_k, x + g, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        x = x.contiguous()
        original_x = x.clone().detach()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight.contiguous()
        if self.gemm.bias is not None:
            b = self.gemm.bias.contiguous()
        else:
            b = torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        # Per-row sum buffer (must be zeroed for atomic adds)
        RowSum = torch.zeros((M,), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
        )
        gemm_rowsum_kernel[grid](
            x, W, b, sub, RowSum,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        out = torch.empty_like(original_x)
        BLOCK_K = 1024
        grid2 = (M, triton.cdiv(K, BLOCK_K))
        gelu_residual_kernel[grid2](original_x, RowSum, out, M, K, N, BLOCK_K=BLOCK_K)

        return out