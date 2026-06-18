import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_persistent_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,  # weight stored as (K, N) row-major: stride_wk=N, stride_wn=1
    SCALE: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Row-stationary persistent kernel: one program per BLOCK_M rows,
    # iterates over all N tiles and tracks running max.
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)

    # Running max across all N tiles
    row_max = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)

    num_n_tiles = tl.cdiv(N, BLOCK_N)

    k0 = 0.7978845608028654
    k1 = 0.044715
    inv_pool = 1.0 / POOL_K

    for tile_n in range(0, num_n_tiles):
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        # weight is K x N: w[k, n] = w_ptr + k*stride_wk + n*stride_wn
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            x = tl.load(x_ptrs, mask=offs_k[None, :] < (K - k), other=0.0)
            w = tl.load(w_ptrs, mask=offs_k[:, None] < (K - k), other=0.0)
            acc += tl.dot(x, w)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]

        # Average pool along N (BLOCK_N must be divisible by POOL_K)
        POOLED_BLOCK_N: tl.constexpr = BLOCK_N // POOL_K
        acc_reshaped = tl.reshape(acc, (BLOCK_M, POOLED_BLOCK_N, POOL_K))
        pooled = tl.sum(acc_reshaped, axis=2) * inv_pool

        # GELU tanh approximation
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
    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)

    fused_persistent_kernel[grid](
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
        # Pre-transpose weight to (K, N) contiguous for efficient B-loads
        self._weight_t_cache = None

    def _get_weight_t(self):
        w = self.matmul.weight  # (N, K)
        # cache transposed contiguous version
        if (self._weight_t_cache is None
                or self._weight_t_cache.data_ptr() == 0
                or self._weight_t_cache.shape != (w.shape[1], w.shape[0])
                or self._weight_t_cache.device != w.device):
            self._weight_t_cache = w.t().contiguous()
        return self._weight_t_cache

    def forward(self, x):
        x = x.cuda()
        weight_t = self._get_weight_t()
        bias = self.matmul.bias.contiguous()
        return fused_forward(x, weight_t, bias, self.pool_kernel_size, self.scale_factor)