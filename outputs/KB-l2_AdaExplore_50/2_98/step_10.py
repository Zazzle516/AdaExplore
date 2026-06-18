import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Fused kernel: GEMM (x @ W^T + b), then average-pool over output dim with kernel size P,
# then GELU(tanh approx), scale, and max-reduce over the pooled dimension.
# Output: (M,) = max over pooled features

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_pool_gelu_scale_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    M, N, K, P, NP,
    SCALE: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # We have 2D grid: (pid_m, pid_pn) where pid_pn iterates over pooled-output tiles
    # Each pid_pn covers BLOCK_N output features = (BLOCK_N // P) pooled outputs
    # We require BLOCK_N % P == 0
    pid_m = tl.program_id(0)
    pid_pn = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_pn * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        # x: (BM, BK), w: (BN, BK), need x @ w.T -> (BM, BN)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # Zero out invalid n positions so they don't contribute (but they shouldn't in pooling)
    acc = tl.where(mask_n[None, :], acc, 0.0)

    # Now do average pool over groups of size P along the N dimension
    # BLOCK_N must be divisible by P; reshape (BM, BN) -> (BM, BN/P, P) -> mean over last
    BN_OVER_P: tl.constexpr = BLOCK_N // P
    acc_reshaped = tl.reshape(acc, (BLOCK_M, BN_OVER_P, P))
    pooled = tl.sum(acc_reshaped, axis=2) / P  # (BM, BN/P)

    # GELU (tanh approx): 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    inner = k0 * (pooled + k1 * pooled * pooled * pooled)
    # tanh via exp
    e2 = tl.exp(2.0 * inner)
    tanh_v = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * pooled * (1.0 + tanh_v)

    # scale
    scaled = gelu * SCALE

    # Mask invalid pooled positions
    pool_offs = pid_pn * BN_OVER_P + tl.arange(0, BN_OVER_P)
    pool_mask = pool_offs < NP
    scaled = tl.where(pool_mask[None, :], scaled, -float('inf'))

    # Reduce max over BN/P -> (BM,) partial max
    partial_max = tl.max(scaled, axis=1)  # (BM,)

    # Atomic max into OUT[m]
    # Use atomic_max on float via reinterpret? Triton has tl.atomic_max for fp32 since recent.
    out_ptrs = OUT_ptr + offs_m
    tl.atomic_max(out_ptrs, partial_max, mask=mask_m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        # Linear layer params (matching nn.Linear init)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        bound = 1.0 / math.sqrt(in_features)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = x.cuda().contiguous()
        W = self.weight.cuda().contiguous()
        b = self.bias.cuda().contiguous()
        M = x.shape[0]
        K = x.shape[1]
        N = W.shape[0]
        P = self.pool_kernel_size
        NP = N // P  # pooled output size (AvgPool1d with no padding floors)

        # Output: max over pooled axis -> shape (M,)
        out = torch.full((M,), float('-inf'), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

        fused_matmul_pool_gelu_scale_kernel[grid](
            x, W, b, out,
            M, N, K, P, NP,
            self.scale_factor,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
        )
        return out