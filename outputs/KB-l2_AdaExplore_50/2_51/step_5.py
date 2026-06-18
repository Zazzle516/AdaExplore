import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Strategy:
# - GEMM produces (B, OUT). Subtract bias-like vector. Mean over OUT -> (B,1).
# - LogSumExp on (B,1) along dim=1 is identity.
# - GELU on (B,1), then add to original_x (B, IN).
#
# Efficient approach: tiled GEMM where each program computes a (BM, BN) tile of the
# output, computes the partial row-sum within the tile (sum over N and add bias-sub)
# and atomically adds that to a per-row partial sum buffer.
# After the GEMM kernel, a small kernel divides by OUT, applies GELU, and we
# do residual add via a fused kernel.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, ROWSUM_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_remaining = K - k0
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # bias - subtract for this N tile
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    sub = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
    bs = bias - sub  # [BLOCK_N]

    acc = acc + bs[None, :]
    acc = tl.where(mask_n[None, :], acc, 0.0)

    # Sum across N tile -> [BLOCK_M]
    partial = tl.sum(acc, axis=1)

    # Atomic add into per-row sum
    tl.atomic_add(ROWSUM_ptr + offs_m, partial, mask=mask_m)


@triton.jit
def gelu_residual_kernel(
    X_ptr, ROWSUM_ptr, OUT_ptr,
    M, F_,
    inv_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    offs = pid_f * BLOCK + tl.arange(0, BLOCK)
    mask = offs < F_

    rs = tl.load(ROWSUM_ptr + pid_m)
    mean_val = rs * inv_N
    # GELU (exact)
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val * inv_sqrt2))

    x_vals = tl.load(X_ptr + pid_m * F_ + offs, mask=mask, other=0.0)
    out_vals = x_vals + gelu_val
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

        rowsum = torch.zeros(B, device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(OUT_F, META['BLOCK_N']),
        )
        fused_gemm_rowsum_kernel[grid](
            x, W, bias, sub, rowsum,
            B, OUT_F, IN_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        out = torch.empty_like(x)
        BLOCK = 1024
        grid2 = (B, (IN_F + BLOCK - 1) // BLOCK)
        gelu_residual_kernel[grid2](
            x, rowsum, out,
            B, IN_F,
            inv_N=1.0 / float(OUT_F),
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out