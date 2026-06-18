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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, PARTIAL_ptr,
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

    # Sum over N tile -> (BLOCK_M,)
    partial = tl.sum(acc, axis=1)

    # Store partial sum to (num_n_tiles, M) buffer at row=pid_n, col=offs_m
    p_ptrs = PARTIAL_ptr + pid_n * stride_pn + offs_m * stride_pm
    tl.store(p_ptrs, partial, mask=mask_m)


@triton.jit
def reduce_partials_kernel(
    PARTIAL_ptr, ROWSUM_ptr,
    M, NUM_N,
    stride_pn, stride_pm,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < NUM_N

    p_ptrs = PARTIAL_ptr + offs_n[:, None] * stride_pn + offs_m[None, :] * stride_pm
    vals = tl.load(p_ptrs, mask=mask_n[:, None] & mask_m[None, :], other=0.0)
    s = tl.sum(vals, axis=0)
    tl.store(ROWSUM_ptr + offs_m, s, mask=mask_m)


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

        # Allocate worst-case partial buffer; we'll size based on a default BLOCK_N=128.
        # Use a conservative upper bound: ceil(OUT_F / 128).
        # Since BLOCK_N varies by autotune config, we allocate based on smallest BLOCK_N (128).
        max_num_n = triton.cdiv(OUT_F, 128)
        partial = torch.empty((max_num_n, B), device=x.device, dtype=torch.float32)
        rowsum = torch.empty(B, device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(OUT_F, META['BLOCK_N']),)
        fused_gemm_rowsum_kernel[grid](
            x, W, bias, sub, partial,
            B, OUT_F, IN_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
        )

        # Determine actual num_n from the chosen config
        best_cfg = fused_gemm_rowsum_kernel.best_config
        actual_block_n = best_cfg.kwargs['BLOCK_N']
        actual_num_n = triton.cdiv(OUT_F, actual_block_n)

        # Reduce partials across N tiles -> rowsum (B,)
        BLOCK_M_RED = 128
        # Pick BLOCK_N_RED as next power of 2 >= actual_num_n
        BLOCK_N_RED = 1
        while BLOCK_N_RED < actual_num_n:
            BLOCK_N_RED *= 2
        BLOCK_N_RED = max(BLOCK_N_RED, 8)
        grid_red = (triton.cdiv(B, BLOCK_M_RED),)
        reduce_partials_kernel[grid_red](
            partial, rowsum,
            B, actual_num_n,
            partial.stride(0), partial.stride(1),
            BLOCK_M=BLOCK_M_RED,
            BLOCK_N=BLOCK_N_RED,
            num_warps=4,
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