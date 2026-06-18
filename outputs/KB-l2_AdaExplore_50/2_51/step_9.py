import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Strategy:
# The reference computes:
#   y = mean_j (W @ x + b - s)_j  -> scalar per row
#   y = logsumexp(y, dim=1) -> still scalar (since dim=1 has size 1)
#   y = gelu(y)
#   out = original_x + y  (broadcast)
#
# We do the GEMM at runtime (safety contract) but fuse the column reduction.
# Specifically: per-row scalar = (1/N) * [ x_i . colsum(W) + sum(b - s) ]
# We compute colsum_W = W.sum(dim=0) at runtime via a Triton kernel,
# then a fused matvec + epilogue + residual add kernel.


@triton.jit
def colsum_w_kernel(
    w_ptr,        # (N, K)
    out_ptr,      # (K,)
    N, K,
    stride_wn, stride_wk,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0,
        )
        acc += tl.sum(w, axis=0)

    tl.store(out_ptr + offs_k, acc, mask=mask_k)


@triton.jit
def matvec_dot_kernel(
    x_ptr,        # (M, K)
    cw_ptr,       # (K,)  colsum(W)
    out_ptr,      # (M,) - per row dot value
    M, K,
    stride_xm, stride_xk,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        ks = k_start + offs_k
        mask_k = ks < K
        x = tl.load(x_ptr + pid_m * stride_xm + ks * stride_xk, mask=mask_k, other=0.0)
        cw = tl.load(cw_ptr + ks, mask=mask_k, other=0.0)
        acc += x * cw
    total = tl.sum(acc, axis=0)
    tl.store(out_ptr + pid_m, total)


@triton.jit
def fused_epilogue_kernel(
    dot_ptr,       # (M,)
    orig_ptr,      # (M, K)
    out_ptr,       # (M, K)
    bs_sum,        # scalar: sum(b - s)
    M, K,
    inv_N,         # 1/N as fp32
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    # load dot value for this row
    d = tl.load(dot_ptr + pid_m)
    mean_val = (d + bs_sum) * inv_N
    # logsumexp over single-element dim = identity
    # GELU (erf form)
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
        original_x = x  # no clone needed; we don't mutate x
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        if self.gemm.bias is not None:
            bs = self.gemm.bias - self.subtract
        else:
            bs = -self.subtract
        bs = bs.contiguous()

        # 1) colsum(W) over N -> (K,)
        colsum = torch.empty((K,), device=x.device, dtype=torch.float32)
        BLOCK_K_CS = 128
        BLOCK_N_CS = 128
        grid_cs = (triton.cdiv(K, BLOCK_K_CS),)
        colsum_w_kernel[grid_cs](
            W, colsum, N, K,
            W.stride(0), W.stride(1),
            BLOCK_N=BLOCK_N_CS, BLOCK_K=BLOCK_K_CS,
            num_warps=4, num_stages=3,
        )

        # 2) per-row dot(x_i, colsum) -> (M,)
        dot_vals = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_K_MV = 1024
        grid_mv = (M,)
        matvec_dot_kernel[grid_mv](
            x, colsum, dot_vals,
            M, K,
            x.stride(0), x.stride(1),
            BLOCK_K=BLOCK_K_MV,
            num_warps=8, num_stages=3,
        )

        # 3) sum(b - s) - small reduction on GPU
        bs_sum = bs.sum()

        # 4) fused epilogue + residual add
        out = torch.empty_like(original_x)
        BLOCK_K_EP = 2048
        grid_ep = (M, triton.cdiv(K, BLOCK_K_EP))
        fused_epilogue_kernel[grid_ep](
            dot_vals, original_x, out,
            bs_sum, M, K,
            1.0 / float(N),
            BLOCK_K=BLOCK_K_EP,
            num_warps=8, num_stages=3,
        )
        return out