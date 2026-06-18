import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_lse_kernel(
    x_ptr, w_ptr, b_ptr,
    partial_max_ptr, partial_sum_ptr,
    M, N, K, NSPLITS,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_pm, stride_ps,
    SCALE2: tl.constexpr,
    CMIN: tl.constexpr,
    CMAX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=mask_m[:, None], other=0.0)
        w = tl.load(w_ptrs, mask=mask_n[None, :], other=0.0)
        acc += tl.dot(x, w)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]
    acc = acc * SCALE2
    acc = tl.minimum(tl.maximum(acc, CMIN), CMAX)

    # Mask invalid N positions with -inf so they don't contribute
    acc = tl.where(mask_n[None, :], acc, -float('inf'))

    # row-wise reduction within this tile
    row_max = tl.max(acc, axis=1)  # (BLOCK_M,)
    row_sum = tl.sum(tl.exp(acc - row_max[:, None]), axis=1)  # (BLOCK_M,)

    # Write to partials: shape (M, NSPLITS)
    out_max_ptrs = partial_max_ptr + offs_m * stride_pm + pid_n * stride_ps
    out_sum_ptrs = partial_sum_ptr + offs_m * stride_pm + pid_n * stride_ps
    tl.store(out_max_ptrs, row_max, mask=mask_m)
    tl.store(out_sum_ptrs, row_sum, mask=mask_m)


@triton.jit
def finalize_lse_mish_kernel(
    partial_max_ptr, partial_sum_ptr, out_ptr,
    M, NSPLITS,
    stride_pm, stride_ps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs = tl.arange(0, BLOCK_S)
    mask = offs < NSPLITS

    maxes = tl.load(partial_max_ptr + pid * stride_pm + offs * stride_ps, mask=mask, other=-float('inf'))
    sums = tl.load(partial_sum_ptr + pid * stride_pm + offs * stride_ps, mask=mask, other=0.0)

    global_max = tl.max(maxes, axis=0)
    # combined sum
    rescaled = sums * tl.exp(maxes - global_max)
    rescaled = tl.where(mask, rescaled, 0.0)
    total_sum = tl.sum(rescaled, axis=0)

    lse = global_max + tl.log(total_sum)

    # x * mish(x) = lse * lse * tanh(softplus(lse))
    sp = tl.log(1.0 + tl.exp(lse))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
    out = lse * lse * tanh_sp

    tl.store(out_ptr + pid, out)


def fused_pipeline(x, weight_kt, bias, scale2, cmin, cmax):
    M, K = x.shape
    K2, N = weight_kt.shape
    assert K == K2

    # We don't know BLOCK_N until autotune picks. Allocate worst-case partials.
    # Use a fixed BLOCK_N grid by leveraging meta lambda.
    # We'll allocate after we know it via a wrapper using preselected block sizes.
    # Strategy: allocate large enough for the smallest BLOCK_N candidate (128 -> 64 splits).
    max_splits = (N + 127) // 128  # smallest BLOCK_N is 128
    partial_max = torch.empty((M, max_splits), device=x.device, dtype=torch.float32)
    partial_sum = torch.empty((M, max_splits), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    # NSPLITS is dynamic; pass it as runtime arg
    # but we need to know actual splits for finalize. Use a trick: compute via cdiv from chosen BLOCK_N.
    # Since BLOCK_N is selected by autotune, we run the kernel which writes only to its splits.
    # For finalize, we need NSPLITS = cdiv(N, BLOCK_N). We'll capture from autotune by using a small helper.
    # Approach: run kernel with grid lambda, then determine BLOCK_N from kernel's best_config.
    fused_gemm_lse_kernel[grid](
        x, weight_kt, bias,
        partial_max, partial_sum,
        M, N, K, max_splits,
        x.stride(0), x.stride(1),
        weight_kt.stride(0), weight_kt.stride(1),
        partial_max.stride(0), partial_max.stride(1),
        SCALE2=float(scale2),
        CMIN=float(cmin),
        CMAX=float(cmax),
    )

    best_cfg = fused_gemm_lse_kernel.best_config
    block_n = best_cfg.kwargs['BLOCK_N']
    actual_splits = (N + block_n - 1) // block_n

    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    # next pow2 >= actual_splits, at least 8
    BLOCK_S = 1
    while BLOCK_S < actual_splits:
        BLOCK_S *= 2
    BLOCK_S = max(BLOCK_S, 8)

    finalize_lse_mish_kernel[(M,)](
        partial_max, partial_sum, out,
        M, actual_splits,
        partial_max.stride(0), partial_max.stride(1),
        BLOCK_S=BLOCK_S,
        num_warps=1,
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
        self.register_buffer('bias_buf', self.matmul.bias.detach().contiguous().cuda())

    def forward(self, x):
        x = x.contiguous().cuda()
        scale2 = self.scale_factor * 2.0
        return fused_pipeline(x, self.weight_kt, self.bias_buf, scale2, self.clamp_min, self.clamp_max)