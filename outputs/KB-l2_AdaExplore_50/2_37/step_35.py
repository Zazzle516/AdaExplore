import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
    ],
    key=['M', 'K'],
)
@triton.jit
def fused_gemm_swish_bias_gn_kernel(
    X_ptr, W_ptr, Lbias_ptr, Bias_ptr, Gamma_ptr, Beta_ptr, Y_ptr,
    M, N: tl.constexpr, K, C: tl.constexpr, G: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles BLOCK_M rows; computes full N=out_features per row
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < M

    # We need to compute Y[row, n] for all n in [0,N) per row, accumulate
    # sum and sum-of-squares per group (group size = C = N/G), then normalize.
    # We'll process N in chunks of size C (channels per group), do GEMM for that
    # chunk, apply swish + bias, accumulate group stats, store to scratch.
    # But we need to keep all N values to re-read for normalization OR do two passes.
    # Strategy: store unnormalized Y to Y_ptr scratch (reuse output buffer),
    # accumulate group stats per row, then in a second loop normalize.

    # Pass 1: compute Y values per group-chunk, write to output, accumulate stats
    # We'll keep group stats in registers: shape [BLOCK_M, G]
    # G=64, BLOCK_M up to 64 -> 4096 floats = manageable
    sum_g = tl.zeros((BLOCK_M, G), dtype=tl.float32)
    sumsq_g = tl.zeros((BLOCK_M, G), dtype=tl.float32)

    # Load X rows once? K=1024, BLOCK_M=64 -> 64K floats = 256KB, too big.
    # Instead loop over K in chunks, and for each output chunk of size C,
    # accumulate. We need to iterate output groups outer, K inner.

    # Outer loop: over groups g in [0, G)
    offs_c = tl.arange(0, C)  # channels within a group
    offs_k = tl.arange(0, BLOCK_K)

    for g in tl.static_range(0, G):
        # output cols are [g*C, g*C + C)
        col_base = g * C
        # accumulator for this group chunk: [BLOCK_M, C]
        acc = tl.zeros((BLOCK_M, C), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_idx = k_start + offs_k
            k_mask = k_idx < K
            # X block: [BLOCK_M, BLOCK_K]
            x_ptrs = X_ptr + rows[:, None] * K + k_idx[None, :]
            x = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
            # W is [N, K] (Linear weight). For output col n, row k: W[n, k]
            # We want W block [BLOCK_K, C] -> W[col, k_idx]^T => shape [BLOCK_K, C]
            w_ptrs = W_ptr + (col_base + offs_c)[None, :] * K + k_idx[:, None]
            w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
            acc += tl.dot(x, w, allow_tf32=True)

        # Add linear bias and extra bias (both shape [N])
        lb = tl.load(Lbias_ptr + col_base + offs_c)
        bb = tl.load(Bias_ptr + col_base + offs_c)
        acc = acc + lb[None, :]
        # Swish: x * sigmoid(x)
        acc = acc * tl.sigmoid(acc)
        # Add bias term
        acc = acc + bb[None, :]

        # Store unnormalized to Y (scratch in output buffer)
        y_ptrs = Y_ptr + rows[:, None] * N + (col_base + offs_c)[None, :]
        tl.store(y_ptrs, acc, mask=row_mask[:, None])

        # Accumulate group stats for this group g
        s = tl.sum(acc, axis=1)  # [BLOCK_M]
        ss = tl.sum(acc * acc, axis=1)  # [BLOCK_M]
        # write into sum_g[:, g], sumsq_g[:, g]
        # Use one-hot mask
        g_mask = tl.arange(0, G) == g
        sum_g = sum_g + s[:, None] * g_mask[None, :].to(tl.float32)
        sumsq_g = sumsq_g + ss[:, None] * g_mask[None, :].to(tl.float32)

    # Compute mean/invstd per group
    inv_c = 1.0 / C
    mean = sum_g * inv_c
    var = sumsq_g * inv_c - mean * mean
    invstd = 1.0 / tl.sqrt(var + EPS)

    # Pass 2: normalize. Re-read Y per group and apply (y - mean) * invstd * gamma + beta
    for g in tl.static_range(0, G):
        col_base = g * C
        cols = col_base + offs_c
        y_ptrs = Y_ptr + rows[:, None] * N + cols[None, :]
        y = tl.load(y_ptrs, mask=row_mask[:, None], other=0.0)
        # gather mean/invstd for this group: sum_g[:, g]
        g_mask = tl.arange(0, G) == g
        m = tl.sum(mean * g_mask[None, :].to(tl.float32), axis=1)
        iv = tl.sum(invstd * g_mask[None, :].to(tl.float32), axis=1)
        gamma = tl.load(Gamma_ptr + cols)
        beta = tl.load(Beta_ptr + cols)
        out = (y - m[:, None]) * iv[:, None] * gamma[None, :] + beta[None, :]
        tl.store(y_ptrs, out, mask=row_mask[:, None])


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        M, K = x.shape
        N = self.out_features
        G = self.num_groups
        C = N // G

        W = self.matmul.weight.contiguous()  # [N, K]
        lb = self.matmul.bias.contiguous()   # [N]
        bb = self.bias.contiguous()          # [N]
        gamma = self.group_norm.weight.contiguous()  # [N]
        beta = self.group_norm.bias.contiguous()     # [N]

        Y = torch.empty((M, N), device=x.device, dtype=x.dtype)

        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
        fused_gemm_swish_bias_gn_kernel[grid](
            x, W, lb, bb, gamma, beta, Y,
            M, N, K, C, G,
            self.eps,
        )
        return Y