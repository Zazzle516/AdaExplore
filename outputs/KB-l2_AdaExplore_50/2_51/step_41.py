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


# Tiled GEMM kernel: computes Y = X @ W^T + b - subtract
# X: [M, K], W: [N, K], b: [N], subtract: [N]
# Y: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sub_kernel(
    X_ptr, W_ptr, b_ptr, sub_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < k_remaining)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    s = tl.load(sub_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :] - s[None, :]

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=y_mask)


# Compute per-row: mean over dim=1, then logsumexp over dim=1 (size 1, so identity),
# then gelu, producing scalar per row.
@triton.jit
def reduce_gelu_kernel(
    Y_ptr, S_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = Y_ptr + pid * N

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_off in range(0, N, BLOCK_N):
        offs = n_off + tl.arange(0, BLOCK_N)
        mask = offs < N
        v = tl.load(row_start + offs, mask=mask, other=0.0)
        acc += v

    s = tl.sum(acc, axis=0) / N
    # logsumexp over a single element is just s
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * s * (1.0 + tl.math.erf(s * inv_sqrt2))
    tl.store(S_ptr + pid, g)


# Fused residual add: out[m, k] = scalar[m] + x[m, k]
@triton.jit
def residual_add_kernel(
    X_ptr, S_ptr, OUT_ptr,
    M, K,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K
    s = tl.load(S_ptr + pid_m)
    x = tl.load(X_ptr + pid_m * K + offs_k, mask=mask, other=0.0)
    out = x + s
    tl.store(OUT_ptr + pid_m * K + offs_k, out, mask=mask)


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

        Y = torch.empty((M, N), device=x.device, dtype=torch.float32)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
        gemm_sub_kernel[grid](
            x, W, b, sub, Y,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            Y.stride(0), Y.stride(1),
        )

        S = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_N = 1024
        reduce_gelu_kernel[(M,)](Y, S, M, N, BLOCK_N=BLOCK_N)

        out = torch.empty_like(original_x)
        BLOCK_K = 1024
        grid2 = (M, triton.cdiv(K, BLOCK_K))
        residual_add_kernel[grid2](original_x, S, out, M, K, BLOCK_K=BLOCK_K)

        return out