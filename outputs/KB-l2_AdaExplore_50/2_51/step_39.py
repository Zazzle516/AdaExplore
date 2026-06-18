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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, PART_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pn, stride_pm,
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

    for k_start in range(0, K, BLOCK_K):
        k_mask = (k_start + offs_k) < K
        x = tl.load(x_ptrs + k_start * stride_xk,
                    mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs + k_start * stride_wk,
                    mask=mask_n[:, None] & k_mask[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    bias_vals = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    sub_vals = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + (bias_vals - sub_vals)[None, :]
    acc = tl.where(mask_n[None, :], acc, 0.0)

    row_partial = tl.sum(acc, axis=1)  # [BLOCK_M]

    # Non-atomic store to partial buffer of shape (num_pid_n, M)
    tl.store(PART_ptr + pid_n * stride_pn + offs_m * stride_pm, row_partial, mask=mask_m)


@triton.jit
def reduce_gelu_kernel(
    PART_ptr, SCALAR_ptr,
    M, NP,
    stride_pn, stride_pm,
    INV_N: tl.constexpr,
    BLOCK_NP: tl.constexpr,
):
    pid = tl.program_id(0)
    # one program per row
    if pid < M:
        offs_np = tl.arange(0, BLOCK_NP)
        mask_np = offs_np < NP
        vals = tl.load(PART_ptr + offs_np * stride_pn + pid * stride_pm,
                       mask=mask_np, other=0.0)
        s = tl.sum(vals, axis=0)
        s = s * INV_N
        inv_sqrt2 = 0.7071067811865475
        g = 0.5 * s * (1.0 + tl.math.erf(s * inv_sqrt2))
        tl.store(SCALAR_ptr + pid, g)


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

        # Estimate max num_pid_n with smallest BLOCK_N in configs (64)
        MAX_NP = (OUT_F + 64 - 1) // 64
        # Partial buffer (NP, B) — we'll only use NP rows actually used
        partial = torch.empty((MAX_NP, B), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(OUT_F, META['BLOCK_N']),
        )
        fused_gemm_rowsum_kernel[grid](
            x, W, bias, sub, partial,
            B, OUT_F, IN_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
        )

        # Determine actual NP used based on selected BLOCK_N
        best_cfg = fused_gemm_rowsum_kernel.best_config
        BLOCK_N_sel = best_cfg.kwargs['BLOCK_N']
        NP = (OUT_F + BLOCK_N_sel - 1) // BLOCK_N_sel

        # Reduce + GELU
        scalar = torch.empty(B, device=x.device, dtype=torch.float32)
        # Find next power of 2 >= NP
        BLOCK_NP = 1
        while BLOCK_NP < NP:
            BLOCK_NP *= 2
        if BLOCK_NP < 16:
            BLOCK_NP = 16
        reduce_gelu_kernel[(B,)](
            partial, scalar,
            B, NP,
            partial.stride(0), partial.stride(1),
            INV_N=1.0 / float(OUT_F),
            BLOCK_NP=BLOCK_NP,
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