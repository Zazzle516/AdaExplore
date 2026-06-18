import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_partial_lse_kernel(
    A_ptr, B_ptr, bias_ptr,
    max_ptr, sumexp_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_max_m, stride_max_n,
    stride_sum_m, stride_sum_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    mask_m = offs_m < M
    mask_n = offs_n < N

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    # Apply mask: set masked-out entries to -inf so they don't affect max/sum
    neg_inf = float('-inf')
    mask_full = mask_m[:, None] & mask_n[None, :]
    acc_masked = tl.where(mask_full, acc, neg_inf)

    # Per-row reduction within this N-tile
    tile_max = tl.max(acc_masked, axis=1)  # [BLOCK_M]
    # If all entries are -inf (e.g., row fully masked), avoid nan
    tile_max_safe = tl.where(tile_max == neg_inf, 0.0, tile_max)
    tile_sumexp = tl.sum(tl.exp(acc_masked - tile_max_safe[:, None]), axis=1)

    # Store to per-(m, n_tile) buffers
    max_out_ptrs = max_ptr + offs_m * stride_max_m + pid_n * stride_max_n
    sum_out_ptrs = sumexp_ptr + offs_m * stride_sum_m + pid_n * stride_sum_n
    tl.store(max_out_ptrs, tile_max, mask=mask_m)
    tl.store(sum_out_ptrs, tile_sumexp, mask=mask_m)


@triton.jit
def final_reduce_kernel(
    max_ptr, sumexp_ptr, out_ptr,
    M, NTILES,
    stride_max_m, stride_max_n,
    stride_sum_m, stride_sum_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return

    offs_n = tl.arange(0, BLOCK_N)
    mask_n = offs_n < NTILES

    neg_inf = float('-inf')
    max_vals = tl.load(max_ptr + pid * stride_max_m + offs_n * stride_max_n,
                       mask=mask_n, other=neg_inf)
    sum_vals = tl.load(sumexp_ptr + pid * stride_sum_m + offs_n * stride_sum_n,
                       mask=mask_n, other=0.0)

    global_max = tl.max(max_vals, axis=0)
    # combine
    adjusted = sum_vals * tl.exp(max_vals - global_max)
    total = tl.sum(adjusted, axis=0)
    lse = global_max + tl.log(total)

    # Apply LeakyReLU twice (slope 0.01)
    x = tl.where(lse >= 0.0, lse, lse * 0.01)
    x = tl.where(x >= 0.0, x, x * 0.01)

    # Apply GELU twice (exact form using erf)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + pid, x)


def fused_forward(x, weight, bias):
    # x: [M, K], weight: [N, K], bias: [N] or None
    M, K = x.shape
    N = weight.shape[0]
    assert weight.shape[1] == K

    x = x.contiguous()
    # B = weight.T => shape [K, N]
    B = weight.t().contiguous()

    has_bias = bias is not None
    if has_bias:
        bias_c = bias.contiguous()
    else:
        bias_c = torch.empty(1, device=x.device, dtype=x.dtype)

    # We need to know N-tile count. Use a fixed BLOCK_N for the partial buffer layout
    # by querying the autotune later isn't trivial; instead, we'll allocate based on a max possible
    # number of tiles, computed using the smallest BLOCK_N in our configs (64).
    # Actually, autotune will pick BLOCK_N at launch, so we need a different approach:
    # We'll allocate partial buffers sized for the chosen config. Simpler: do a fixed grid using
    # a meta lambda for the grid, and size the buffer using the same meta.

    # Pre-allocate buffers using max tiles (assume min BLOCK_N = 64)
    MIN_BLOCK_N = 64
    max_ntiles = triton.cdiv(N, MIN_BLOCK_N)

    max_buf = torch.full((M, max_ntiles), float('-inf'), device=x.device, dtype=torch.float32)
    sum_buf = torch.zeros((M, max_ntiles), device=x.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))

    gemm_partial_lse_kernel[grid](
        x, B, bias_c,
        max_buf, sum_buf,
        M, N, K,
        x.stride(0), x.stride(1),
        B.stride(0), B.stride(1),
        max_buf.stride(0), max_buf.stride(1),
        sum_buf.stride(0), sum_buf.stride(1),
        HAS_BIAS=has_bias,
    )

    # Determine actual ntiles used by chosen config
    # We can't easily know post-launch, so just reduce across all max_ntiles columns
    # (unused columns are -inf for max and 0 for sum which is safe).
    out = torch.empty((M, 1), device=x.device, dtype=torch.float32)

    # Pick BLOCK_N power of 2 >= max_ntiles
    block_n = 1
    while block_n < max_ntiles:
        block_n *= 2
    block_n = max(block_n, 16)

    final_reduce_kernel[(M,)](
        max_buf, sum_buf, out,
        M, max_ntiles,
        max_buf.stride(0), max_buf.stride(1),
        sum_buf.stride(0), sum_buf.stride(1),
        BLOCK_N=block_n,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(ModelNew, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        x = x.cuda()
        weight = self.linear.weight
        bias = self.linear.bias if self.linear.bias is not None else None
        return fused_forward(x, weight, bias)