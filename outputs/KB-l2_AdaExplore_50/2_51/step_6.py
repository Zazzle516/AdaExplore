import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Strategy:
# We need to compute s_b = mean_n( sum_k W[n,k]*x[b,k] + bias[n] - subtract[n] ) for each batch b.
# Then GELU(s_b) and broadcast-add to original x.
#
# Tiled GEMM approach: tile over (M=batch, N=out_features). Each program computes
# a (BLOCK_M, BLOCK_N) tile of (X @ W^T + bias - subtract), reduces along N within
# tile, then atomically adds row partial sums to a (B,) accumulator.
#
# Then a small kernel divides by OUT_F, applies GELU, and we fuse residual-add
# into a final elementwise kernel.


@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, ROWSUM_ptr,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    # X tile pointer: [BLOCK_M, BLOCK_K]
    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W tile pointer: [BLOCK_N, BLOCK_K]  -> we want W^T multiplied, so do tl.dot(x, w.T)
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x = tl.load(x_ptrs + k_start * stride_xk,
                    mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs + k_start * stride_wk,
                    mask=mask_n[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    # Add bias - subtract
    bias_vals = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    sub_vals = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + (bias_vals - sub_vals)[None, :]

    # Mask invalid n
    acc = tl.where(mask_n[None, :], acc, 0.0)

    # Reduce along N
    row_partial = tl.sum(acc, axis=1)  # [BLOCK_M]

    # Atomic add into rowsum
    tl.atomic_add(ROWSUM_ptr + offs_m, row_partial, mask=mask_m)


@triton.jit
def gelu_scalar_kernel(
    ROWSUM_ptr, OUT_ptr,
    M, INV_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    v = tl.load(ROWSUM_ptr + offs, mask=mask, other=0.0)
    v = v * INV_N
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * v * (1.0 + tl.math.erf(v * inv_sqrt2))
    tl.store(OUT_ptr + offs, g, mask=mask)


@triton.jit
def residual_add_kernel(
    X_ptr, S_ptr, OUT_ptr,
    M, F_,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs = pid_f * BLOCK + tl.arange(0, BLOCK)
    mask = offs < F_

    x_vals = tl.load(X_ptr + pid_m * F_ + offs, mask=mask, other=0.0)
    s_val = tl.load(S_ptr + pid_m)
    out_vals = x_vals + s_val
    tl.store(OUT_ptr + pid_m * F_ + offs, out_vals, mask=mask)


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
        IN_F = self.in_features
        OUT_F = self.out_features

        W = self.gemm.weight.contiguous()  # (out, in)
        if self.gemm.bias is not None:
            bias = self.gemm.bias.contiguous()
        else:
            bias = torch.zeros(OUT_F, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        # Row sum accumulator
        rowsum = torch.zeros(B, device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        grid = ((B + BLOCK_M - 1) // BLOCK_M, (OUT_F + BLOCK_N - 1) // BLOCK_N)
        fused_gemm_rowsum_kernel[grid](
            x, W, bias, sub, rowsum,
            B, OUT_F, IN_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=3,
        )

        # Apply mean (divide by OUT_F) + GELU
        scalar = torch.empty(B, device=x.device, dtype=torch.float32)
        BLOCK_S = 256
        grid_s = ((B + BLOCK_S - 1) // BLOCK_S,)
        gelu_scalar_kernel[grid_s](
            rowsum, scalar,
            B, INV_N=1.0 / float(OUT_F),
            BLOCK=BLOCK_S,
            num_warps=2,
        )

        # Residual add
        out = torch.empty_like(x)
        BLOCK = 1024
        grid2 = (B, (IN_F + BLOCK - 1) // BLOCK)
        residual_add_kernel[grid2](
            x, scalar, out,
            B, IN_F,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out