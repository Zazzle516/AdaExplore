import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The key insight: after gemm -> subtract -> mean(dim=1) -> logsumexp(dim=1) -> gelu,
# we get a [batch_size, 1] tensor. Each element is a scalar function of one row.
# 
# For a row x of input:
#   y = gemm(x) - subtract   -> shape [out_features]
#   m = mean(y) = (sum(W @ x) + sum(b) - sum(subtract)) / out_features
#   lse(m) over dim=1 with size 1 = m  (logsumexp of single element is itself)
#   g = gelu(m)
#   out = g + original_x  (broadcast scalar to [in_features])
#
# But we must execute every op at runtime per safety contract. So we compute
# gemm fully, then do the reductions and elementwise ops with fused kernels.


# Tiled GEMM kernel: computes Y = X @ W^T + b - subtract, fused with per-row sum reduction
# X: [M, K], W: [N, K] (stride_wk=1 contiguous), b: [N], subtract: [N]
# RowSum: [M] - atomically accumulated sum of each row of (X@W^T + b - sub)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_sub_rowsum_kernel(
    X_ptr, Wt_ptr, b_ptr, sub_ptr, Partial_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wtk, stride_wtn,
    stride_pm, stride_pt,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    EVEN_K: tl.constexpr,
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

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    # Wt: [K, N] contiguous, stride_wtn = 1
    w_ptrs = Wt_ptr + (offs_k[:, None] * stride_wtk + offs_n[None, :] * stride_wtn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_K:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            x = tl.load(x_ptrs)
            w = tl.load(w_ptrs)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wtk
    else:
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_remaining = K - k * BLOCK_K
            x_mask = offs_k[None, :] < k_remaining
            w_mask = offs_k[:, None] < k_remaining
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)
            w = tl.load(w_ptrs, mask=w_mask, other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wtk

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    s = tl.load(sub_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :] - s[None, :]

    n_mask = offs_n < N
    acc = tl.where(n_mask[None, :], acc, 0.0)

    row_partial = tl.sum(acc, axis=1)  # [BLOCK_M]

    m_mask = offs_m < M
    # Write partial sum to Partial[offs_m, pid_n]
    p_ptrs = Partial_ptr + offs_m * stride_pm + pid_n * stride_pt
    tl.store(p_ptrs, row_partial, mask=m_mask)


# Reduce partial row-sums [M, NUM_TILES] -> [M] then mean -> gelu.
@triton.jit
def reduce_mean_gelu_kernel(
    Partial_ptr, S_ptr,
    M, N, NUM_TILES,
    stride_pm, stride_pt,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < M:
        offs_t = tl.arange(0, BLOCK_T)
        mask = offs_t < NUM_TILES
        p = tl.load(Partial_ptr + pid * stride_pm + offs_t * stride_pt, mask=mask, other=0.0)
        rs = tl.sum(p, axis=0)
        s = rs / N
        inv_sqrt2 = 0.7071067811865475
        g = 0.5 * s * (1.0 + tl.math.erf(s * inv_sqrt2))
        tl.store(S_ptr + pid, g)


# Fused residual add: out[m, k] = scalar[m] + x[m, k]
@triton.jit
def residual_add_kernel(
    X_ptr, S_ptr, OUT_ptr,
    M, K,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = offs_k < K
    s = tl.load(S_ptr + pid_m)
    x = tl.load(X_ptr + pid_m * K + offs_k, mask=mask, other=0.0)
    out = x + s
    tl.store(OUT_ptr + pid_m * K + offs_k, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        self._Wt_cache = None
        self._Wt_version = -1

    def _get_Wt(self):
        W = self.gemm.weight
        if self._Wt_cache is None or self._Wt_version != W._version or self._Wt_cache.device != W.device:
            self._Wt_cache = W.t().contiguous()
            self._Wt_version = W._version
        return self._Wt_cache

    def forward(self, x):
        x = x.contiguous()
        original_x = x.clone().detach()
        M, K = x.shape
        N = self.out_features

        Wt = self._get_Wt()  # [K, N] contiguous
        if self.gemm.bias is not None:
            b = self.gemm.bias.contiguous()
        else:
            b = torch.zeros(N, device=x.device, dtype=x.dtype)
        sub = self.subtract.contiguous()

        # Determine num_n_tiles from the autotuner's chosen BLOCK_N. We use a max bound
        # then truncate via mask. Use upper bound = ceil(N / 64) to cover smallest BLOCK_N.
        # Actually we need the exact number of tiles, which depends on autotune choice.
        # Solution: allocate worst-case Partial sized [M, max_tiles], pass NUM_TILES from meta.
        max_tiles = triton.cdiv(N, 64)  # smallest BLOCK_N in configs is 64
        Partial = torch.empty((M, max_tiles), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

        # EVEN_K: K divisible by all BLOCK_K candidates (32, 64) — true when K % 64 == 0
        even_k = (K % 64 == 0)

        gemm_sub_rowsum_kernel[grid](
            x, Wt, b, sub, Partial,
            M, N, K,
            x.stride(0), x.stride(1),
            Wt.stride(0), Wt.stride(1),
            Partial.stride(0), Partial.stride(1),
            EVEN_K=even_k,
        )

        # Get the chosen BLOCK_N from autotuner
        chosen = gemm_sub_rowsum_kernel.best_config
        BLOCK_N_chosen = chosen.kwargs['BLOCK_N']
        num_tiles = triton.cdiv(N, BLOCK_N_chosen)

        S = torch.empty((M,), device=x.device, dtype=torch.float32)
        # BLOCK_T must be a power of 2 >= num_tiles
        BLOCK_T = 1
        while BLOCK_T < num_tiles:
            BLOCK_T *= 2
        BLOCK_T = max(BLOCK_T, 16)
        reduce_mean_gelu_kernel[(M,)](
            Partial, S, M, N, num_tiles,
            Partial.stride(0), Partial.stride(1),
            BLOCK_T=BLOCK_T,
        )

        out = torch.empty_like(original_x)
        BLOCK_K = 1024
        grid2 = (M, triton.cdiv(K, BLOCK_K))
        residual_add_kernel[grid2](original_x, S, out, M, K, BLOCK_K=BLOCK_K)

        return out