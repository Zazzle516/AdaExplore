import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, BS_ptr, ROWSUM_ptr,
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

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs + k_start * stride_xk)
        w = tl.load(w_ptrs + k_start * stride_wk)
        acc += tl.dot(x, w)

    bs_vals = tl.load(BS_ptr + offs_n)
    acc = acc + bs_vals[None, :]

    row_partial = tl.sum(acc, axis=1)
    tl.atomic_add(ROWSUM_ptr + offs_m, row_partial)


@triton.jit
def gelu_residual_add_kernel(
    X_ptr, ROWSUM_ptr, OUT_ptr,
    M, F_,
    INV_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs = pid_f * BLOCK + tl.arange(0, BLOCK)
    mask = offs < F_

    rs = tl.load(ROWSUM_ptr + pid_m)
    v = rs * INV_N
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * v * (1.0 + tl.math.erf(v * inv_sqrt2))

    x_vals = tl.load(X_ptr + pid_m * F_ + offs, mask=mask, other=0.0)
    out_vals = x_vals + g
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

        W = self.gemm.weight.contiguous()
        if self.gemm.bias is not None:
            bias = self.gemm.bias
        else:
            bias = torch.zeros(OUT_F, device=x.device, dtype=x.dtype)
        sub = self.subtract

        bs = (bias - sub).contiguous()

        rowsum = torch.zeros(B, device=x.device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32

        grid = ((B + BLOCK_M - 1) // BLOCK_M, (OUT_F + BLOCK_N - 1) // BLOCK_N)
        fused_gemm_rowsum_kernel[grid](
            x, W, bs, rowsum,
            B, OUT_F, IN_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=4,
        )

        out = torch.empty_like(x)
        BLOCK = 2048
        grid2 = (B, (IN_F + BLOCK - 1) // BLOCK)
        gelu_residual_add_kernel[grid2](
            x, rowsum, out,
            B, IN_F,
            INV_N=1.0 / float(OUT_F),
            BLOCK=BLOCK,
            num_warps=8,
        )

        return out