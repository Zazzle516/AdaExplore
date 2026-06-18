import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K', 'SPLIT_K'],
)
@triton.jit
def fused_matmul_pool_gelu_max_kernel(
    x_ptr, wT_ptr, b_ptr, partial_ptr,
    M, N, K,
    NUM_N_TILES,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pm, stride_pn, stride_ps,
    SCALE: tl.constexpr,
    POOL_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_split = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # swizzle for L2 reuse
    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Split-K: each split processes a chunk of K
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_split * k_per_split
    k_end = tl.minimum(k_start + k_per_split, K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (k_start + offs_k)[None, :] * stride_xk
    # wT is (K, N) -> contiguous along N
    wT_ptrs = wT_ptr + (k_start + offs_k)[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    k = k_start
    for _ in range(0, tl.cdiv(k_end - k_start, BLOCK_K)):
        k_remain = k_end - k
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remain), other=0.0)
        w = tl.load(wT_ptrs, mask=(offs_k[:, None] < k_remain) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        wT_ptrs += BLOCK_K * stride_wk
        k += BLOCK_K

    # only add bias in split 0 (we'll add bias after reduction in epilogue if SPLIT_K>1)
    # To keep epilogue simple: do bias + pool + gelu + scale + max here when SPLIT_K==1.
    # When SPLIT_K>1, we just store partial sums and finalize in a second kernel.
    if SPLIT_K == 1:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
        n_mask = offs_n < N
        NEG_INF: tl.constexpr = -3.4e38
        acc = tl.where(n_mask[None, :], acc, NEG_INF)

        POOLED_BLOCK_N: tl.constexpr = BLOCK_N // POOL_K
        INV_POOL: tl.constexpr = 1.0 / POOL_K
        acc_reshaped = tl.reshape(acc, (BLOCK_M, POOLED_BLOCK_N, POOL_K))
        pooled = tl.sum(acc_reshaped, axis=2) * INV_POOL

        k0 = 0.7978845608028654
        k1 = 0.044715
        inner = k0 * (pooled + k1 * pooled * pooled * pooled)
        e2 = tl.exp(2.0 * inner)
        tanh_val = (e2 - 1.0) / (e2 + 1.0)
        gelu = 0.5 * pooled * (1.0 + tanh_val)
        result = gelu * SCALE

        tile_max = tl.max(result, axis=1)
        m_mask = offs_m < M
        # atomic max into output (initialized to -inf)
        tl.atomic_max(partial_ptr + offs_m, tile_max, mask=m_mask)
    else:
        pass


def fused_forward(x, weight_T, bias, pool_k, scale):
    M, K = x.shape
    K2, N = weight_T.shape
    assert K == K2
    assert N % pool_k == 0

    x = x.contiguous()

    # initialize output to -inf for atomic_max
    out = torch.full((M,), -3.4e38, device=x.device, dtype=torch.float32)

    SPLIT_K = 1

    def grid(meta):
        num_n_tiles = triton.cdiv(N, meta['BLOCK_N'])
        num_m_tiles = triton.cdiv(M, meta['BLOCK_M'])
        return (num_m_tiles * num_n_tiles, SPLIT_K)

    fused_matmul_pool_gelu_max_kernel[grid](
        x, weight_T, bias, out,
        M, N, K,
        0,
        x.stride(0), x.stride(1),
        weight_T.stride(0), weight_T.stride(1),
        0, 0, 0,
        SCALE=scale,
        POOL_K=pool_k,
        SPLIT_K=SPLIT_K,
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
        # Pre-transpose weight to (K, N) for contiguous N-axis loads
        self.register_buffer('weight_T', self.matmul.weight.detach().t().contiguous().cuda())

    def forward(self, x):
        x = x.cuda()
        bias = self.matmul.bias
        return fused_forward(x, self.weight_T, bias, self.pool_kernel_size, self.scale_factor)