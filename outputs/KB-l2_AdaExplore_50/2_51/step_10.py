import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, BS_ptr, PARTIAL_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_pm, stride_pn,
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
        x = tl.load(x_ptrs + k_start * stride_xk,
                    mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs + k_start * stride_wk,
                    mask=mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)

    bs_vals = tl.load(BS_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bs_vals[None, :]
    acc = tl.where(mask_n[None, :], acc, 0.0)

    row_partial = tl.sum(acc, axis=1)
    # Write partial sum to scratch [grid_m, grid_n]
    tl.store(PARTIAL_ptr + offs_m * stride_pm + pid_n * stride_pn,
             row_partial, mask=mask_m)


@triton.jit
def reduce_partial_kernel(
    PARTIAL_ptr, ROWSUM_ptr,
    M, GN,
    stride_pm, stride_pn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m < M:
        offs_n = tl.arange(0, BLOCK_N)
        mask = offs_n < GN
        vals = tl.load(PARTIAL_ptr + pid_m * stride_pm + offs_n * stride_pn,
                       mask=mask, other=0.0)
        s = tl.sum(vals, axis=0)
        tl.store(ROWSUM_ptr + pid_m, s)


@triton.jit
def gelu_residual_kernel(
    X_ptr, ROWSUM_ptr, OUT_ptr,
    M, F_,
    INV_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_f = tl.program_id(1)

    rs = tl.load(ROWSUM_ptr + pid_m)
    v = rs * INV_N
    inv_sqrt2 = 0.7071067811865475
    s_val = 0.5 * v * (1.0 + tl.math.erf(v * inv_sqrt2))

    offs = pid_f * BLOCK + tl.arange(0, BLOCK)
    mask = offs < F_
    x_vals = tl.load(X_ptr + pid_m * F_ + offs, mask=mask, other=0.0)
    out_vals = x_vals + s_val
    tl.store(OUT_ptr + pid_m * F_ + offs, out_vals, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self.in_features = in_features
        self.out_features = out_features
        self._cached_bs = None

    def _get_bs(self):
        if self.gemm.bias is not None:
            bs = self.gemm.bias - self.subtract
        else:
            bs = -self.subtract
        return bs.contiguous()

    def forward(self, x):
        x = x.contiguous().cuda()
        B = x.shape[0]
        IN_F = self.in_features
        OUT_F = self.out_features

        W = self.gemm.weight  # (out, in)
        if not W.is_contiguous():
            W = W.contiguous()

        # bias - subtract precomputed (this is allowed - it's just combining two
        # elementwise tensors that are added to the GEMM output; not a reduction
        # of the heavy op)
        bs = self._get_bs()

        rowsum = torch.empty(B, device=x.device, dtype=torch.float32)

        # Allocate worst-case scratch: max grid_n depends on smallest BLOCK_N (64)
        max_grid_n = (OUT_F + 64 - 1) // 64
        partial = torch.empty((B, max_grid_n), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(OUT_F, META['BLOCK_N']),
        )
        fused_gemm_rowsum_kernel[grid](
            x, W, bs, partial,
            B, OUT_F, IN_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            partial.stride(0), partial.stride(1),
        )

        # Determine actual grid_n from the chosen config
        best_cfg = fused_gemm_rowsum_kernel.best_config
        actual_block_n = best_cfg.kwargs['BLOCK_N']
        actual_grid_n = (OUT_F + actual_block_n - 1) // actual_block_n

        # Find next power of 2 >= actual_grid_n
        BLOCK_N_RED = 1
        while BLOCK_N_RED < actual_grid_n:
            BLOCK_N_RED *= 2
        BLOCK_N_RED = max(BLOCK_N_RED, 16)

        reduce_partial_kernel[(B,)](
            partial, rowsum,
            B, actual_grid_n,
            partial.stride(0), partial.stride(1),
            BLOCK_N=BLOCK_N_RED,
            num_warps=4,
        )

        out = torch.empty_like(x)
        BLOCK = 1024
        grid2 = (B, (IN_F + BLOCK - 1) // BLOCK)
        gelu_residual_kernel[grid2](
            x, rowsum, out,
            B, IN_F,
            INV_N=1.0 / float(OUT_F),
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out