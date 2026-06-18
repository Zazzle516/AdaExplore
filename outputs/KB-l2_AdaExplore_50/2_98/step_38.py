import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_pool_gelu_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K, P,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_op,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_mask = offs_m < M
    n_mask = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        x = tl.load(x_ptrs, mask=(m_mask[:, None]) & (offs_k[None, :] < k_remaining), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_remaining) & (n_mask[None, :]), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # acc has shape (BLOCK_M, BLOCK_N), where BLOCK_N is multiple of POOL
    # Reshape to (BLOCK_M, BLOCK_N // POOL, POOL) and reduce-mean over POOL
    NUM_POOL = BLOCK_N // POOL
    acc_r = tl.reshape(acc, (BLOCK_M, NUM_POOL, POOL))
    pooled = tl.sum(acc_r, axis=2) / POOL  # (BLOCK_M, NUM_POOL)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu_val = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))
    scaled = gelu_val * SCALE

    # Store to Out at (offs_m, pid_n * NUM_POOL + range(NUM_POOL))
    p_start = pid_n * NUM_POOL
    offs_p = p_start + tl.arange(0, NUM_POOL)
    p_mask = offs_p < P

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_p[None, :] * stride_op
    store_mask = m_mask[:, None] & p_mask[None, :]
    tl.store(out_ptrs, scaled, mask=store_mask)


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

        lin = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(lin.weight.detach().clone())
        self.bias = nn.Parameter(lin.bias.detach().clone())

        assert out_features % pool_kernel_size == 0
        self.pooled_size = out_features // pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        W = self.weight.contiguous()
        B = self.bias.contiguous()

        M, K = x.shape
        N = self.out_features
        POOL = self.pool_kernel_size
        P = self.pooled_size

        intermediate = torch.empty((M, P), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

        fused_gemm_pool_gelu_kernel[grid](
            x, W, B, intermediate,
            M, N, K, P,
            x.stride(0), x.stride(1),
            W.stride(0), W.stride(1),
            intermediate.stride(0), intermediate.stride(1),
            POOL=POOL,
            SCALE=self.scale_factor,
        )

        out = torch.empty((M,), device=x.device, dtype=torch.float32)
        BLOCK_P = 1
        while BLOCK_P < P:
            BLOCK_P *= 2

        row_max_kernel[(M,)](
            intermediate, out,
            M, P,
            BLOCK_P=BLOCK_P,
        )

        return out