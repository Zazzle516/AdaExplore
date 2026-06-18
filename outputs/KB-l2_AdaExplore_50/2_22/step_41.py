import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_clamp_partial_lse_kernel(
    x_ptr, w_ptr, b_ptr,
    partial_max_ptr, partial_sum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pm, stride_pn,
    SCALE2: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
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
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = acc * SCALE2
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    # Mask out-of-range columns with -inf so they don't affect max/sum
    neg_inf = float('-inf')
    acc_masked = tl.where(mask_n[None, :], acc, neg_inf)

    # Row-wise reduction within this tile
    tile_max = tl.max(acc_masked, axis=1)  # (BLOCK_M,)
    # sum exp(x - tile_max). For rows where tile_max=-inf, this would be 0.
    # All rows here are valid (mask_m), but tile could be all-invalid columns (won't happen since mask_n is true for in-range tiles)
    diff = acc_masked - tile_max[:, None]
    e = tl.exp(diff)
    e = tl.where(mask_n[None, :], e, 0.0)
    tile_sum = tl.sum(e, axis=1)  # (BLOCK_M,)

    # Store partial (max, sum) at position (m_block, n_block)
    p_offs = offs_m * stride_pm + pid_n * stride_pn
    tl.store(partial_max_ptr + p_offs, tile_max, mask=mask_m)
    tl.store(partial_sum_ptr + p_offs, tile_sum, mask=mask_m)


@triton.jit
def reduce_lse_mish_kernel(
    partial_max_ptr, partial_sum_ptr, out_ptr,
    M, NB,
    stride_pm, stride_pn,
    BLOCK_NB: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs = tl.arange(0, BLOCK_NB)
    mask = offs < NB

    pm = tl.load(partial_max_ptr + pid * stride_pm + offs * stride_pn, mask=mask, other=float('-inf'))
    ps = tl.load(partial_sum_ptr + pid * stride_pm + offs * stride_pn, mask=mask, other=0.0)

    row_max = tl.max(pm, axis=0)
    # sum_i ps_i * exp(pm_i - row_max)
    contrib = ps * tl.exp(pm - row_max)
    contrib = tl.where(mask, contrib, 0.0)
    total = tl.sum(contrib, axis=0)
    lse = row_max + tl.log(total)

    # mish(lse) = lse * tanh(softplus(lse))
    sp = tl.log(1.0 + tl.exp(lse))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    mish_val = lse * tanh_sp
    out = lse * mish_val

    tl.store(out_ptr + pid, out)


def fused_pipeline(x, weight_kt, bias, scale2, cmin, cmax):
    M, K = x.shape
    K2, N = weight_kt.shape
    assert K == K2

    # We need to know BLOCK_N to size partial buffers. Use a fixed worst-case via autotune meta.
    # Easiest: allocate partial buffers sized by max possible NB (N / smallest BLOCK_N=64).
    # Better: do it after autotune resolves — use grid lambda that allocates.
    # We'll allocate inside grid lambda by storing in a closure.

    partials = {}

    def grid(meta):
        BLOCK_M = meta['BLOCK_M']
        BLOCK_N = meta['BLOCK_N']
        NB = triton.cdiv(N, BLOCK_N)
        MB = triton.cdiv(M, BLOCK_M)
        partials['NB'] = NB
        partials['BLOCK_M'] = BLOCK_M
        partials['BLOCK_N'] = BLOCK_N
        return (MB * NB,)

    # We don't know NB until autotune picks config. Use a two-phase: allocate the maximum.
    # Allocate generous max: assume BLOCK_N >= 64 -> NB <= N/64
    NB_max = triton.cdiv(N, 64)
    partial_max = torch.empty((M, NB_max), device=x.device, dtype=torch.float32)
    partial_sum = torch.empty((M, NB_max), device=x.device, dtype=torch.float32)

    fused_gemm_clamp_partial_lse_kernel[grid](
        x, weight_kt, bias,
        partial_max, partial_sum,
        M, N, K,
        x.stride(0), x.stride(1),
        weight_kt.stride(0), weight_kt.stride(1),
        partial_max.stride(0), partial_max.stride(1),
        SCALE2=float(scale2),
        CMIN=float(cmin),
        CMAX=float(cmax),
    )

    NB = partials['NB']

    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    # Find smallest power of two >= NB
    BLOCK_NB = 1
    while BLOCK_NB < NB:
        BLOCK_NB *= 2
    BLOCK_NB = max(BLOCK_NB, 16)

    reduce_lse_mish_kernel[(M,)](
        partial_max, partial_sum, out,
        M, NB,
        partial_max.stride(0), partial_max.stride(1),
        BLOCK_NB=BLOCK_NB,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.register_buffer('weight_kt', self.matmul.weight.detach().t().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        bias = self.matmul.bias.contiguous()
        scale2 = self.scale_factor * 2.0
        out = fused_pipeline(x, self.weight_kt, bias, scale2, self.clamp_min, self.clamp_max)
        return out