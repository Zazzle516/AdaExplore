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
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_pool_gelu_max_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    M, N, K,
    NUM_N_TILES,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_remain = K - k
        x = tl.load(x_ptrs, mask=offs_k[None, :] < k_remain, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[None, :] < k_remain, other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    POOLED_BLOCK_N: tl.constexpr = BLOCK_N // POOL_K
    acc_reshaped = tl.reshape(acc, (BLOCK_M, POOLED_BLOCK_N, POOL_K))
    pooled = tl.sum(acc_reshaped, axis=2) * (1.0 / POOL_K)

    k0 = 0.7978845608028654
    k1 = 0.044715
    inner = k0 * (pooled + k1 * pooled * pooled * pooled)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * pooled * (1.0 + tanh_val)
    result = gelu * SCALE

    tile_max = tl.max(result, axis=1)

    m_mask = offs_m < M
    p_ptrs = partial_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(p_ptrs, tile_max, mask=m_mask)


@triton.jit
def row_max_reduce_kernel(
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


def fused_forward(x, weight, bias, pool_k, scale):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    assert N % pool_k == 0

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((M,), device=x.device, dtype=torch.float32)

    MAX_NUM_N_TILES = triton.cdiv(N, 128)
    partial = torch.empty((M, MAX_NUM_N_TILES), device=x.device, dtype=torch.float32)

    def grid(meta):
        num_n_tiles = triton.cdiv(N, meta['BLOCK_N'])
        num_m_tiles = triton.cdiv(M, meta['BLOCK_M'])
        return (num_m_tiles * num_n_tiles,)

    fused_matmul_pool_gelu_max_kernel[grid](
        x, weight, bias, partial,
        M, N, K,
        MAX_NUM_N_TILES,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        partial.stride(0), partial.stride(1),
        SCALE=scale,
        POOL_K=pool_k,
    )

    best_conf = fused_matmul_pool_gelu_max_kernel.best_config
    actual_block_n = best_conf.kwargs['BLOCK_N']
    actual_num_n_tiles = triton.cdiv(N, actual_block_n)

    BLOCK_R = triton.next_power_of_2(actual_num_n_tiles)
    if BLOCK_R < 16:
        BLOCK_R = 16
    row_max_reduce_kernel[(M,)](
        partial, out,
        M, actual_num_n_tiles,
        partial.stride(0), partial.stride(1),
        BLOCK_N=BLOCK_R,
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