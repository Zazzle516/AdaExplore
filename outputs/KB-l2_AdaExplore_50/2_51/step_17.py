import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# After GEMM, subtract, mean over dim=1 -> shape (B,1), logsumexp over dim=1 -> shape (B,1) (no-op since size 1),
# GELU -> shape (B,1), then add original_x (B, in_features) -> broadcast.
# So the whole post-GEMM tail collapses to: per-row scalar = GELU(mean(linear(x) - subtract))
# Then output = scalar[:, None] + original_x
# We must still execute the GEMM at runtime.

# Strategy:
# 1) Compute y = x @ W^T + b via a tiled GEMM kernel. We only need the per-row sum of (y - subtract)
#    to compute the mean. But per safety contract, every operator must execute at runtime - we should
#    not collapse the GEMM into a matvec by precomputing a column sum of W.
#    So we'll actually compute the full y matrix in a Triton GEMM, then do mean reduce.
# Actually re-reading: "Do not collapse a heavy op into a downstream reduction at init time
#  (for example, precomputing a column/row sum of a weight so that forward runs a matvec
#   instead of the full operator)". The key phrase is "at init time". We should run the full GEMM.

# We'll do:
# - Triton GEMM kernel producing y (B, out) with bias and -subtract fused
# - Triton reduction kernel computing mean per row -> scalar (B,)
# - Apply GELU on scalar
# - Triton kernel adding scalar to original_x

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_sub_rowsum_kernel(
    X_ptr, W_ptr, B_ptr, S_ptr, ROWSUM_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M

    row_sum = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_mask_x = (offs_k[None, :] + k) < K
            k_mask_w = (offs_k[:, None] + k) < K
            x = tl.load(x_ptrs, mask=mask_m[:, None] & k_mask_x, other=0.0)
            w = tl.load(w_ptrs, mask=k_mask_w & mask_n[None, :], other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
        s = tl.load(S_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + b[None, :] - s[None, :]
        acc = tl.where(mask_n[None, :], acc, 0.0)
        row_sum += tl.sum(acc, axis=1)

    tl.store(ROWSUM_ptr + offs_m, row_sum, mask=mask_m)


@triton.jit
def rowsum_to_gelu_kernel(
    ROWSUM_ptr, OUT_ptr,
    M, N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < M
    s = tl.load(ROWSUM_ptr + offs, mask=mask, other=0.0)
    mean = s / N
    x = mean
    gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
    tl.store(OUT_ptr + offs, gelu, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def add_scalar_rowwise_kernel(
    X_ptr, S_ptr, OUT_ptr,
    M, N,
    BLOCK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs = pid_n * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X_ptr + pid_m * N + offs, mask=mask, other=0.0)
    s = tl.load(S_ptr + pid_m)
    out = x + s
    tl.store(OUT_ptr + pid_m * N + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        original_x = x
        if not x.is_contiguous():
            x = x.contiguous()
        B, K = x.shape
        N = self.out_features
        W = self.gemm.weight  # (N, K)
        bias = self.gemm.bias if self.gemm.bias is not None else torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract

        rowsum = torch.empty((B,), device=x.device, dtype=torch.float32)

        grid = lambda META: (triton.cdiv(B, META['BLOCK_M']),)
        gemm_bias_sub_rowsum_kernel[grid](
            x, W, bias, sub, rowsum,
            B, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )

        scalar = torch.empty((B,), device=x.device, dtype=x.dtype)
        BLOCK = 256
        rowsum_to_gelu_kernel[(triton.cdiv(B, BLOCK),)](rowsum, scalar, B, N, BLOCK=BLOCK)

        out = torch.empty_like(x)
        grid2 = lambda META: (B, triton.cdiv(K, META['BLOCK']))
        add_scalar_rowwise_kernel[grid2](x, scalar, out, B, K)
        return out