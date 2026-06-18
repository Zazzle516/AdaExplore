import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_lse_kernel(
    x_ptr, w_ptr, b_ptr, partial_max_ptr, partial_sum_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Load W as [BLOCK_N, BLOCK_K] with K axis contiguous (stride_wk=1)
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        k_remaining = K - k
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = acc * SCALE2
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    # Mask off invalid n positions with -inf for max reduction
    acc_for_max = tl.where(mask_n[None, :], acc, -float('inf'))
    tile_max = tl.max(acc_for_max, axis=1)  # [BLOCK_M]

    # sum of exp(acc - tile_max), zeroing invalid positions
    exp_vals = tl.exp(acc - tile_max[:, None])
    exp_vals = tl.where(mask_n[None, :], exp_vals, 0.0)
    tile_sum = tl.sum(exp_vals, axis=1)  # [BLOCK_M]

    # Store partial max and sum
    pm_ptrs = partial_max_ptr + offs_m * stride_pm + pid_n * stride_pn
    ps_ptrs = partial_sum_ptr + offs_m * stride_pm + pid_n * stride_pn
    tl.store(pm_ptrs, tile_max, mask=mask_m)
    tl.store(ps_ptrs, tile_sum, mask=mask_m)


@triton.jit
def reduce_lse_mish_kernel(
    partial_max_ptr, partial_sum_ptr, out_ptr,
    M, NUM_TILES,
    stride_pm, stride_pn,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs = tl.arange(0, BLOCK_T)
    mask = offs < NUM_TILES

    pm = tl.load(partial_max_ptr + pid * stride_pm + offs * stride_pn,
                 mask=mask, other=-float('inf'))
    ps = tl.load(partial_sum_ptr + pid * stride_pm + offs * stride_pn,
                 mask=mask, other=0.0)

    global_max = tl.max(pm, axis=0)
    # rescale
    rescaled = ps * tl.exp(pm - global_max)
    rescaled = tl.where(mask, rescaled, 0.0)
    total = tl.sum(rescaled, axis=0)
    lse = global_max + tl.log(total)

    # x * mish(x) where x=lse: lse * lse * tanh(softplus(lse))
    sp = tl.log(1.0 + tl.exp(lse))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = lse * lse * tanh_sp

    tl.store(out_ptr + pid, out)


def fused_forward(x, weight, bias, scale2, cmin, cmax):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    # We need to know BLOCK_N to size partials, but autotune picks it.
    # Use a max possible NUM_TILES based on smallest BLOCK_N (64), then truncate.
    # Strategy: allocate partials of shape (M, ceil(N/64)) which is upper bound.
    MAX_TILES = (N + 63) // 64
    partial_max = torch.empty((M, MAX_TILES), device=x.device, dtype=torch.float32)
    partial_sum = torch.empty((M, MAX_TILES), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    fused_gemm_lse_kernel[grid](
        x, weight, bias, partial_max, partial_sum,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        partial_max.stride(0), partial_max.stride(1),
        SCALE2=float(scale2),
        CMIN=float(cmin),
        CMAX=float(cmax),
    )

    # Get actual BLOCK_N used to know NUM_TILES
    best_config = fused_gemm_lse_kernel.best_config
    BLOCK_N = best_config.kwargs['BLOCK_N']
    NUM_TILES = (N + BLOCK_N - 1) // BLOCK_N

    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

    # next power of 2 >= NUM_TILES
    BLOCK_T = 1
    while BLOCK_T < NUM_TILES:
        BLOCK_T *= 2
    if BLOCK_T < 16:
        BLOCK_T = 16

    reduce_lse_mish_kernel[(M,)](
        partial_max, partial_sum, out,
        M, NUM_TILES,
        partial_max.stride(0), partial_max.stride(1),
        BLOCK_T=BLOCK_T,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, scale_factor, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.matmul = nn.Linear(input_size, hidden_size)
        self.scale_factor = scale_factor
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.matmul.weight.contiguous()
        bias = self.matmul.bias.contiguous()
        scale2 = self.scale_factor * 2.0
        return fused_forward(x, weight, bias, scale2, self.clamp_min, self.clamp_max)