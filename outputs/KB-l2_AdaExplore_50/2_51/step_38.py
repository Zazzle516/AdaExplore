import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_rowsum_kernel(
    X_ptr, W_ptr, BS_ptr, SCRATCH_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_sn, stride_sm,
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
    tl.store(SCRATCH_ptr + pid_n * stride_sn + offs_m * stride_sm, row_partial)


@triton.jit
def reduce_scratch_kernel(
    SCRATCH_ptr, ROWSUM_ptr,
    M, NUM_TILES,
    stride_sn, stride_sm,
    BLOCK_T: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_t = tl.arange(0, BLOCK_T)
    mask = offs_t < NUM_TILES
    vals = tl.load(SCRATCH_ptr + offs_t * stride_sn + pid_m * stride_sm, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)
    tl.store(ROWSUM_ptr + pid_m, s)


@triton.jit
def gelu_residual_add_kernel(
    X_ptr, ROWSUM_ptr, OUT_ptr,
    M, F_,
    INV_N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)

    rs = tl.load(ROWSUM_ptr + pid_m)
    v = rs * INV_N
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * v * (1.0 + tl.math.erf(v * inv_sqrt2))

    base = pid_m * F_
    for off_start in tl.static_range(0, 1):
        offs = tl.arange(0, BLOCK)
        # Process the row in chunks of BLOCK
        pass

    # Process the row - assuming F_ is a multiple of BLOCK
    num_chunks = F_ // BLOCK
    for i in range(num_chunks):
        offs = i * BLOCK + tl.arange(0, BLOCK)
        x_vals = tl.load(X_ptr + base + offs)
        out_vals = x_vals + g
        tl.store(OUT_ptr + base + offs, out_vals)


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

        # Allocate scratch buffer for partial sums - shape (num_pid_n_max, B)
        # We don't know BLOCK_N yet (autotuned), so we allocate for the smallest tile
        max_num_pid_n = (OUT_F + 64 - 1) // 64
        scratch = torch.empty((max_num_pid_n, B), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            triton.cdiv(B, META['BLOCK_M']) * triton.cdiv(OUT_F, META['BLOCK_N']),
        )
        fused_gemm_rowsum_kernel[grid](
            x, W, bs, scratch,
            B, OUT_F, IN_F,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            scratch.stride(0), scratch.stride(1),
        )

        # Determine the actual BLOCK_N used and reduce
        best_config = fused_gemm_rowsum_kernel.best_config
        actual_block_n = best_config.kwargs['BLOCK_N']
        actual_num_tiles = (OUT_F + actual_block_n - 1) // actual_block_n

        rowsum = torch.empty(B, device=x.device, dtype=torch.float32)
        # Find next power of 2 >= actual_num_tiles
        BLOCK_T = 1
        while BLOCK_T < actual_num_tiles:
            BLOCK_T *= 2
        BLOCK_T = max(BLOCK_T, 16)

        reduce_scratch_kernel[(B,)](
            scratch, rowsum,
            B, actual_num_tiles,
            scratch.stride(0), scratch.stride(1),
            BLOCK_T=BLOCK_T,
            num_warps=2,
        )

        out = torch.empty_like(x)
        BLOCK = 1024
        if IN_F % BLOCK != 0:
            BLOCK = IN_F
        grid2 = (B,)
        gelu_residual_add_kernel[grid2](
            x, rowsum, out,
            B, IN_F,
            INV_N=1.0 / float(OUT_F),
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )

        return out