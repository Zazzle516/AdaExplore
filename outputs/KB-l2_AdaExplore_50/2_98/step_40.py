import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_pool_gelu_max_kernel(
    x_ptr, wt_ptr, b_ptr, out_ptr,
    M, N, K,  # M=batch, N=out_features, K=in_features
    stride_xm, stride_xk,
    stride_wk, stride_wn,  # weight is (K, N) layout
    SCALE: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Persistent over N tiles for one M tile.
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Running per-row max
    NEG_INF = float('-inf')
    row_max = tl.full((BLOCK_M,), NEG_INF, dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)

    for nt in range(0, num_n_tiles):
        offs_n = nt * BLOCK_N + tl.arange(0, BLOCK_N)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = wt_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_remain = K - k
            x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remain), other=0.0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_remain) & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        # Add bias
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

        n_mask = offs_n < N
        acc = tl.where(n_mask[None, :], acc, 0.0)

        # Average pool along N
        POOLED_BLOCK_N: tl.constexpr = BLOCK_N // POOL_K
        acc_reshaped = tl.reshape(acc, (BLOCK_M, POOLED_BLOCK_N, POOL_K))
        pooled = tl.sum(acc_reshaped, axis=2) * (1.0 / POOL_K)

        # GELU (tanh approx)
        k0 = 0.7978845608028654
        k1 = 0.044715
        inner = k0 * (pooled + k1 * pooled * pooled * pooled)
        e2 = tl.exp(2.0 * inner)
        tanh_val = (e2 - 1.0) / (e2 + 1.0)
        gelu = 0.5 * pooled * (1.0 + tanh_val)
        result = gelu * SCALE

        tile_max = tl.max(result, axis=1)
        row_max = tl.maximum(row_max, tile_max)

    m_mask = offs_m < M
    tl.store(out_ptr + offs_m, row_max, mask=m_mask)


def fused_forward(x, weight_t, bias, pool_k, scale):
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
    assert N % pool_k == 0

    x = x.contiguous()
    bias = bias.contiguous()

    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)

    fused_matmul_pool_gelu_max_kernel[grid](
        x, weight_t, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        SCALE=scale,
        POOL_K=pool_k,
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
        self._weight_t_cache = None

    def _get_weight_t(self):
        w = self.matmul.weight
        if (self._weight_t_cache is None
                or self._weight_t_cache.data_ptr() == 0
                or self._weight_t_cache.device != w.device
                or self._weight_t_cache.shape[0] != w.shape[1]):
            self._weight_t_cache = w.t().contiguous()
        return self._weight_t_cache

    def forward(self, x):
        x = x.cuda()
        if not self.matmul.weight.is_cuda:
            self.matmul = self.matmul.cuda()
            self._weight_t_cache = None
        weight_t = self._get_weight_t()
        bias = self.matmul.bias
        return fused_forward(x, weight_t, bias, self.pool_kernel_size, self.scale_factor)