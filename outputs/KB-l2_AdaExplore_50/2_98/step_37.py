import torch
import torch.nn as nn
import triton
import triton.language as tl
import math

# Fused kernel: computes y = x @ W^T + b, then for each output row:
#   - groups of `pool_kernel_size` consecutive outputs are averaged
#   - GELU applied, scaled, then max-reduced across pooled dimension
# One program per (batch_row, pooled_block_tile) — but we do full row in one program for simplicity.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_K': 256}, num_warps=8, num_stages=3),
    ],
    key=['K', 'N', 'POOL'],
)
@triton.jit
def fused_matmul_pool_gelu_scale_max_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_N: tl.constexpr,   # = POOL, number of output features per pooled element
    BLOCK_K: tl.constexpr,
):
    # program_id(0) = batch row
    # program_id(1) = pooled output index (along N // POOL)
    pid_m = tl.program_id(0)
    pid_p = tl.program_id(1)

    # Compute POOL output features [n_start, n_start+POOL)
    n_start = pid_p * POOL
    offs_n = n_start + tl.arange(0, BLOCK_N)  # BLOCK_N == POOL

    # Accumulator for POOL output features
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Pointers to X row and W rows
    x_row_ptr = X_ptr + pid_m * stride_xm  # shape (K,)
    # W is (N, K), each output feature n has weight W[n, :]
    # We want to compute for each n in offs_n: dot(X[m,:], W[n,:])

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        # Load x: shape (BLOCK_K,)
        x_vals = tl.load(x_row_ptr + offs_k * stride_xk, mask=k_mask, other=0.0)
        # Load W: shape (BLOCK_N, BLOCK_K)
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (k_mask[None, :])
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # acc += sum over k of w * x
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    # Add bias
    b_vals = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b_vals

    # Average pool: mean of the POOL elements
    pooled = tl.sum(acc, axis=0) / POOL  # scalar

    # GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))

    # Scale
    scaled = gelu_val * SCALE

    # Store to intermediate buffer at (m, pid_p)
    # Out is (M, P) where P = N // POOL
    P = N // POOL
    tl.store(Out_ptr + pid_m * P + pid_p, scaled)


@triton.jit
def row_max_kernel(
    In_ptr, Out_ptr,
    M, P,
    BLOCK_P: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_P)
    mask = offs < P
    vals = tl.load(In_ptr + pid * P + offs, mask=mask, other=-float('inf'))
    m = tl.max(vals, axis=0)
    tl.store(Out_ptr + pid, m)


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)

        # Use nn.Linear to match parameter initialization
        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())

        assert out_features % pool_kernel_size == 0, "out_features must be divisible by pool_kernel_size"
        self.pooled_size = out_features // pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.weight.contiguous()
        B = self.bias.contiguous()

        M, K = x.shape
        N = self.out_features
        POOL = self.pool_kernel_size
        P = self.pooled_size

        # Intermediate buffer for pooled+gelu+scale results: (M, P)
        intermediate = torch.empty((M, P), device=x.device, dtype=torch.float32)

        grid = (M, P)
        fused_matmul_pool_gelu_scale_max_kernel[grid](
            x, W, B, intermediate,
            M, N, K,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            POOL=POOL,
            SCALE=self.scale_factor,
            BLOCK_N=POOL,
        )

        # Now reduce max along P dimension
        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        # pick BLOCK_P as next power of 2 >= P
        BLOCK_P = 1
        while BLOCK_P < P:
            BLOCK_P *= 2

        row_max_kernel[(M,)](
            intermediate, out,
            M, P,
            BLOCK_P=BLOCK_P,
        )

        return out