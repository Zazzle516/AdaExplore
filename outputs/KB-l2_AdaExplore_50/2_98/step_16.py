import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_pool_gelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,  # M=batch, N=out_features, K=in_features
    POOLED_N,  # N // pool_kernel
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    SCALE: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # BLOCK_N must be a multiple of POOL_K
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remain), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] < k_remain), other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    # Mask out-of-bounds N to 0 (so they don't contribute to pooled avg incorrectly)
    n_mask = offs_n < N
    acc = tl.where(n_mask[None, :], acc, 0.0)

    # Average pool along N: reshape BLOCK_N into (BLOCK_N // POOL_K, POOL_K) and reduce
    POOLED_BLOCK_N: tl.constexpr = BLOCK_N // POOL_K
    acc_reshaped = tl.reshape(acc, (BLOCK_M, POOLED_BLOCK_N, POOL_K))
    pooled = tl.sum(acc_reshaped, axis=2) / POOL_K  # (BLOCK_M, POOLED_BLOCK_N)

    # GELU (tanh approximation)
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    k0 = 0.7978845608028654  # sqrt(2/pi)
    k1 = 0.044715
    inner = k0 * (pooled + k1 * pooled * pooled * pooled)
    # tanh via exp
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * pooled * (1.0 + tanh_val)

    # Scale
    result = gelu * SCALE

    # Store
    pooled_offs_n = pid_n * POOLED_BLOCK_N + tl.arange(0, POOLED_BLOCK_N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + pooled_offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (pooled_offs_n[None, :] < POOLED_N)
    tl.store(out_ptrs, result, mask=out_mask)


@triton.jit
def row_max_kernel(
    inp_ptr, out_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    offs_n = tl.arange(0, BLOCK_N)
    neg_inf = float('-inf')
    max_val = tl.full((), neg_inf, dtype=tl.float32)
    for n_start in range(0, N, BLOCK_N):
        cur = n_start + offs_n
        mask = cur < N
        vals = tl.load(inp_ptr + pid * stride_m + cur * stride_n, mask=mask, other=neg_inf)
        chunk_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, chunk_max)
    tl.store(out_ptr + pid, max_val)


def fused_forward(x, weight, bias, pool_k, scale):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    assert N % pool_k == 0
    POOLED_N = N // pool_k

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    pooled = torch.empty((M, POOLED_N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    fused_matmul_pool_gelu_kernel[grid](
        x, weight, bias, pooled,
        M, N, K,
        POOLED_N,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        pooled.stride(0), pooled.stride(1),
        SCALE=scale,
        POOL_K=pool_k,
    )

    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    BLOCK_N = 1024
    row_max_kernel[(M,)](
        pooled, out,
        M, POOLED_N,
        pooled.stride(0), pooled.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, pool_kernel_size, scale_factor):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pool_kernel_size = pool_kernel_size
        self.scale_factor = float(scale_factor)
        self.matmul = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = x.cuda()
        weight = self.matmul.weight
        bias = self.matmul.bias
        return fused_forward(x, weight, bias, self.pool_kernel_size, self.scale_factor)