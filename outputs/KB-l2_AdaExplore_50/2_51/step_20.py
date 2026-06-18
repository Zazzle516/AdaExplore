import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Strategy:
# - Run the GEMM as a real matmul at runtime (cuBLAS via torch.mm) to honor
#   the safety contract: the heavy operator must execute on the actual input.
# - Fuse the post-GEMM epilogue:
#     row_scalar_m = (1/N) * sum_j (gemm[m,j] - subtract[j] + bias[j])
#     gelu_val = gelu(row_scalar_m)            # logsumexp over size-1 dim is identity
#     out[m, k] = original_x[m, k] + gelu_val
#   into two small Triton kernels: a row-reduce that produces gelu_val per row,
#   and a residual-add kernel that writes the final output.


@triton.jit
def row_reduce_gelu_kernel(
    gemm_ptr,      # (M, N)
    bs_ptr,        # (N,)  = bias - subtract  (or -subtract if no bias)
    out_ptr,       # (M,)  per-row gelu(mean)
    M, N,
    inv_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    offs = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        ns = n_start + offs
        mask = ns < N
        g = tl.load(gemm_ptr + pid_m * N + ns, mask=mask, other=0.0)
        b = tl.load(bs_ptr + ns, mask=mask, other=0.0)
        acc += (g + b)

    total = tl.sum(acc, axis=0)
    mean_val = total * inv_N
    # GELU (erf form)
    inv_sqrt2 = 0.70710678118654752440
    gelu_val = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val * inv_sqrt2))
    tl.store(out_ptr + pid_m, gelu_val)


@triton.jit
def residual_add_kernel(
    orig_ptr,      # (M, K)
    scalar_ptr,    # (M,)
    out_ptr,       # (M, K)
    M, K,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    s = tl.load(scalar_ptr + pid_m)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    orig = tl.load(orig_ptr + pid_m * K + offs_k, mask=mask_k, other=0.0)
    out = orig + s
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
        original_x = x
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        if self.gemm.bias is not None:
            bs = self.gemm.bias.detach() - self.subtract.detach()
        else:
            bs = -self.subtract.detach()
        bs = bs.contiguous()

        # 1) Real GEMM via cuBLAS: gemm_out = x @ W.T  -> (M, N)
        gemm_out = torch.mm(x, W.t())

        # 2) Fused per-row reduction + GELU
        row_scalar = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_N = 1024
        grid_r = (M,)
        row_reduce_gelu_kernel[grid_r](
            gemm_out, bs, row_scalar,
            M, N,
            inv_N=1.0 / float(N),
            BLOCK_N=BLOCK_N,
            num_warps=8, num_stages=3,
        )

        # 3) Residual add (broadcast scalar per row)
        out = torch.empty_like(original_x)
        BLOCK_K = 2048
        grid_a = (M, triton.cdiv(K, BLOCK_K))
        residual_add_kernel[grid_a](
            original_x, row_scalar, out,
            M, K,
            BLOCK_K=BLOCK_K,
            num_warps=8, num_stages=2,
        )
        return out