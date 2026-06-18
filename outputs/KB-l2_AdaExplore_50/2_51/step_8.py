import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The key insight: after GEMM, subtract, mean over dim=1, logsumexp over dim=1 (which is size 1),
# the result is a scalar per batch row. logsumexp of a single element is just that element.
# So x becomes shape (B, 1) = mean of (gemm(x) - subtract) over out_features.
# Then GELU, then add to original (B, in_features) -> broadcasts.
#
# mean over dim=1 of (W @ x + b - s) = (1/N) * sum_j (sum_k W[j,k]*x[k] + b[j] - s[j])
#                                    = (1/N) * (sum_k x[k] * sum_j W[j,k] + sum_j(b[j]-s[j]))
# But safety contract says no graph-level shortcuts that fold reductions into weights.
# So we must actually do the GEMM at runtime.
#
# Strategy: do a fused kernel that computes per-row sum of (W @ x + b - s),
# i.e. for each row i, compute sum_j (sum_k W[j,k] * x[i,k]) + sum_j(b[j]-s[j]).
# We compute the full GEMM tile by tile and reduce along j.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_rowsum_kernel(
    x_ptr, w_ptr, bs_ptr,  # x:(M,K), w:(N,K) (linear weight), bs:(N,) = bias - subtract
    partial_ptr,            # (M, num_n_blocks) - partial sums over N tiles
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_n_blocks = tl.num_programs(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K
        x = tl.load(x_ptrs + k * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs + k * stride_wk, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    # add (bias - subtract) per column
    bs = tl.load(bs_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bs[None, :]
    # zero out invalid columns
    acc = tl.where(mask_n[None, :], acc, 0.0)
    # reduce along N within this tile
    row_partial = tl.sum(acc, axis=1)  # (BLOCK_M,)

    out_ptrs = partial_ptr + offs_m * num_n_blocks + pid_n
    tl.store(out_ptrs, row_partial, mask=mask_m)


@triton.jit
def finalize_kernel(
    partial_ptr,   # (M, num_n_blocks)
    orig_ptr,      # (M, K)
    out_ptr,       # (M, K)
    M, K, NUM_NB, N,
    BLOCK_K: tl.constexpr,
    NUM_NB_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    if pid_m >= M:
        return

    # load partials and sum
    nb_offs = tl.arange(0, NUM_NB_C)
    nb_mask = nb_offs < NUM_NB
    parts = tl.load(partial_ptr + pid_m * NUM_NB + nb_offs, mask=nb_mask, other=0.0)
    total = tl.sum(parts, axis=0)
    mean_val = total / N
    # logsumexp of single element = itself
    # GELU
    # use erf-based GELU
    inv_sqrt2 = 0.70710678118654752440
    gelu_val = 0.5 * mean_val * (1.0 + tl.math.erf(mean_val * inv_sqrt2))

    # residual add
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_k = offs_k < K
    orig = tl.load(orig_ptr + pid_m * K + offs_k, mask=mask_k, other=0.0)
    out = orig + gelu_val
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
        original_x = x.clone()
        M, K = x.shape
        N = self.out_features

        W = self.gemm.weight  # (N, K)
        if self.gemm.bias is not None:
            bs = self.gemm.bias - self.subtract
        else:
            bs = -self.subtract
        bs = bs.contiguous()

        # Determine grid
        # We'll use autotuned block sizes; pre-allocate partial with max num_n_blocks
        # Use a fixed BLOCK_N choice for allocation: compute after autotune chooses.
        # Workaround: query autotune by running a small calc — easier: allocate based on lambda meta.

        def grid_gemm(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        # We need partial sized (M, num_n_blocks). Since BLOCK_N is chosen by autotune,
        # allocate based on smallest possible BLOCK_N to be safe. We'll use a callback approach:
        # Instead, fix the partial shape by computing num_n_blocks for each call. Use a dict cache.

        # Simpler: use a wrapper that allocates partial after autotuner chose.
        # Triton's autotune selects config before kernel runs, but the launch already needs `partial_ptr`.
        # Solution: allocate the max-needed size. Smallest BLOCK_N in configs is 64.
        min_block_n = 64
        max_num_nb = (N + min_block_n - 1) // min_block_n
        partial = torch.empty((M, max_num_nb), device=x.device, dtype=torch.float32)

        # We need to know actual num_n_blocks used. Use a hook via meta in grid.
        chosen = {}

        def grid(meta):
            nb = triton.cdiv(N, meta['BLOCK_N'])
            chosen['nb'] = nb
            chosen['bn'] = meta['BLOCK_N']
            return (triton.cdiv(M, meta['BLOCK_M']), nb)

        # The partial layout uses stride = num_n_blocks per row. We must match this in finalize.
        # We pass num_n_blocks at launch. Since partial is over-allocated with max_num_nb columns,
        # but kernel writes to `offs_m * num_n_blocks + pid_n`. We must use the ACTUAL num_n_blocks
        # as the row stride. So allocate exactly the right size after we know it.
        # Trick: compute num_n_blocks for each candidate config. We'll just pick a config manually
        # to avoid this complexity.

        # Manual config selection (skip autotune complexity):
        BLOCK_M = 32
        BLOCK_N = 128
        BLOCK_K = 64
        num_nb = (N + BLOCK_N - 1) // BLOCK_N
        partial = torch.empty((M, num_nb), device=x.device, dtype=torch.float32)

        grid_manual = (triton.cdiv(M, BLOCK_M), num_nb)

        gemm_rowsum_kernel_manual[grid_manual](
            x, W, bs, partial,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        out = torch.empty_like(original_x)
        BLOCK_K_FIN = 256
        NUM_NB_C = triton.next_power_of_2(num_nb)
        grid_fin = (M, triton.cdiv(K, BLOCK_K_FIN))
        finalize_kernel[grid_fin](
            partial, original_x, out,
            M, K, num_nb, N,
            BLOCK_K=BLOCK_K_FIN, NUM_NB_C=NUM_NB_C,
            num_warps=4,
        )
        return out


@triton.jit
def gemm_rowsum_kernel_manual(
    x_ptr, w_ptr, bs_ptr,
    partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    num_n_blocks = tl.num_programs(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K
        x = tl.load(x_ptrs + k * stride_xk, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs + k * stride_wk, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))

    bs = tl.load(bs_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bs[None, :]
    acc = tl.where(mask_n[None, :], acc, 0.0)
    row_partial = tl.sum(acc, axis=1)

    out_ptrs = partial_ptr + offs_m * num_n_blocks + pid_n
    tl.store(out_ptrs, row_partial, mask=mask_m)