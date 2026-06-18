import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=5),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_pool_gelu_max_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K,
    NUM_N_TILES,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pm, stride_pn,
    SCALE: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # weight is transposed: shape (K, N)
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

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

    # Mask out-of-bounds N to -inf (so they don't affect max)
    n_mask = offs_n < N
    neg_inf = float('-inf')
    # For pool we need to mask to 0, so handle pool first using 0-mask
    acc_pool = tl.where(n_mask[None, :], acc, 0.0)

    # Average pool along N
    POOLED_BLOCK_N: tl.constexpr = BLOCK_N // POOL_K
    acc_reshaped = tl.reshape(acc_pool, (BLOCK_M, POOLED_BLOCK_N, POOL_K))
    pooled = tl.sum(acc_reshaped, axis=2) / POOL_K

    # GELU (tanh approx)
    k0 = 0.7978845608028654
    k1 = 0.044715
    inner = k0 * (pooled + k1 * pooled * pooled * pooled)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * pooled * (1.0 + tanh_val)
    result = gelu * SCALE

    # Mask out-of-bounds pooled positions
    pooled_offs_n = pid_n * POOLED_BLOCK_N + tl.arange(0, POOLED_BLOCK_N)
    POOLED_N = N // POOL_K
    pooled_mask = pooled_offs_n < POOLED_N
    result = tl.where(pooled_mask[None, :], result, neg_inf)

    # Reduce max across pooled tile dim
    tile_max = tl.max(result, axis=1)  # (BLOCK_M,)

    # Store partial max
    m_mask = offs_m < M
    partial_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(partial_ptrs, tile_max, mask=m_mask)


@triton.jit
def final_max_kernel(
    inp_ptr, out_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    mask = offs_n < N
    vals = tl.load(inp_ptr + pid * stride_m + offs_n * stride_n, mask=mask, other=float('-inf'))
    m = tl.max(vals, axis=0)
    tl.store(out_ptr + pid, m)


def fused_forward(x, weight_t, bias, pool_k, scale):
    M, K = x.shape
    K2, N = weight_t.shape
    assert K == K2
    assert N % pool_k == 0

    x = x.contiguous()
    bias = bias.contiguous()

    # Allocate partial max tensor; size depends on BLOCK_N chosen by autotuner.
    # Worst-case: smallest BLOCK_N in configs is 128 -> max num_n_tiles = N/128
    MAX_NUM_N_TILES = triton.cdiv(N, 128)
    partial = torch.empty((M, MAX_NUM_N_TILES), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    fused_matmul_pool_gelu_max_kernel[grid](
        x, weight_t, bias, partial,
        M, N, K,
        MAX_NUM_N_TILES,
        x.stride(0), x.stride(1),
        weight_t.stride(0), weight_t.stride(1),
        partial.stride(0), partial.stride(1),
        SCALE=scale,
        POOL_K=pool_k,
    )

    # Get actual num_n_tiles used
    best_cfg = fused_matmul_pool_gelu_max_kernel.best_config
    block_n = best_cfg.kwargs['BLOCK_N']
    num_n_tiles = triton.cdiv(N, block_n)

    out = torch.empty((M,), device=x.device, dtype=torch.float32)
    # Pick next power-of-2 for BLOCK_N reduction
    bn = 1
    while bn < num_n_tiles:
        bn *= 2
    bn = max(bn, 8)
    final_max_kernel[(M,)](
        partial, out,
        M, num_n_tiles,
        partial.stride(0), partial.stride(1),
        BLOCK_N=bn,
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
                or self._weight_t_cache.device != w.device):
            self._weight_t_cache = w.t().contiguous()
        return self._weight_t_cache

    def forward(self, x):
        x = x.cuda()
        weight_t = self._get_weight_t()
        bias = self.matmul.bias
        return fused_forward(x, weight_t, bias, self.pool_kernel_size, self.scale_factor)