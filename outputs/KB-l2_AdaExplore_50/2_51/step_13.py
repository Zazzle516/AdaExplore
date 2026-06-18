import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Strategy:
# Forward computes:
#   y = Linear(x) - subtract           # (B, out)
#   m = mean(y, dim=1, keepdim=True)   # (B, 1)
#   l = logsumexp(m, dim=1, keepdim=True)  # (B,1) == m (single element)
#   g = gelu(l)                         # (B, 1)
#   out = g + original_x                # (B, in_features)
#
# We must execute the GEMM at runtime (safety contract).
# Use a tiled GEMM kernel with tl.dot: programs tile over (M, N).
# Each program computes its (BLOCK_M, BLOCK_N) tile of (X @ W^T + bias - subtract),
# performs a partial sum over its N tile (per-row), and atomic-adds to a per-row
# accumulator buffer of shape (B,). After the GEMM kernel finishes, a small
# elementwise kernel divides by OUT_F, applies GELU, and adds to original_x.


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

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is (N, K), accessed as W[offs_n, offs_k]; transpose for tl.dot
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        mask_k = offs_k < k_remaining
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x_tile, w_tile)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias_vals = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    sub_vals = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + (bias_vals - sub_vals)[None, :]
    acc = tl.where(mask_m[:, None] & mask_n[None, :], acc, 0.0)

    # Sum over N tile -> (BLOCK_M,)
    partial = tl.sum(acc, axis=1)

    # Atomic add to rowsum buffer
    tl.atomic_add(ROWSUM_ptr + offs_m, partial, mask=mask_m)


@triton.jit
def gelu_residual_kernel(
    ROWSUM_ptr, ORIG_ptr, OUT_ptr,
    M, F_DIM,
    inv_N,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs_f = pid_f * BLOCK + tl.arange(0, BLOCK)
    mask_f = offs_f < F_DIM

    rs = tl.load(ROWSUM_ptr + pid_m)
    mean_val = rs * inv_N
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val * inv_sqrt2))

    x_vals = tl.load(ORIG_ptr + pid_m * F_DIM + offs_f, mask=mask_f, other=0.0)
    out_vals = x_vals + g
    tl.store(OUT_ptr + pid_m * F_DIM + offs_f, out_vals, mask=mask_f)


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

        rowsum = torch.zeros(B, device=x.device, dtype=torch.float32)

        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        grid = (triton.cdiv(B, BLOCK_M), triton.cdiv(OUT_F, BLOCK_N))
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

        out = torch.empty_like(x)
        BLOCK_F = 1024
        grid2 = (B, triton.cdiv(IN_F, BLOCK_F))
        inv_N = 1.0 / float(OUT_F)
        gelu_residual_kernel[grid2](
            rowsum, x, out,
            B, IN_F,
            inv_N,
            BLOCK=BLOCK_F,
            num_warps=4,
        )

        return out