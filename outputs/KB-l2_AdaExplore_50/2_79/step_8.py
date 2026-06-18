import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_kernel(
    x_ptr,           # [N, C, S]  conv output
    mult_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    clamp_min,
    clamp_max,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, s_block)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # First: for instance norm, need mean and var of (x * mult) over S, per (n, c).
    # We'll compute them in a separate kernel approach; but since we need them per
    # (n,c) and we already loop over s here, use a two-pass within this program:
    # Actually each program covers a tile of S, so we can't compute full reduction here.
    # We split into two kernels. This kernel assumes mean/invstd are precomputed.
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def stats_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    mean_ptr,        # [N, C]
    invstd_ptr,      # [N, C]
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # n * C + c
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)

    # accumulate sum and sumsq over S
    sum_acc = 0.0
    sumsq_acc = 0.0

    base = n * C * S + c * S
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        sum_acc += tl.sum(tl.where(mask, y, 0.0), axis=0)
        sumsq_acc += tl.sum(tl.where(mask, y * y, 0.0), axis=0)

    mean = sum_acc / S
    var = sumsq_acc / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    mean_ptr,        # [N, C]
    invstd_ptr,      # [N, C]
    out_ptr,         # [N, S]
    N, C, S,
    clamp_min,
    clamp_max,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)

    # Load multipliers, mean, invstd: shape [BLOCK_C]
    # Assume BLOCK_C == C (padded exactly), no mask needed
    m = tl.load(mult_ptr + c_offs)
    mean = tl.load(mean_ptr + pid_n * C + c_offs)
    invstd = tl.load(invstd_ptr + pid_n * C + c_offs)

    # Load x[n, :, s] shape [BLOCK_C, BLOCK_S]
    x_ptrs = x_ptr + pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    x = tl.load(x_ptrs, mask=s_mask[None, :], other=0.0)

    # y = x * m
    y = x * m[:, None]
    # normalize
    y = (y - mean[:, None]) * invstd[:, None]
    # clamp
    y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
    # multiply by m again
    y = y * m[:, None]

    # max over c (no mask since BLOCK_C == C)
    out = tl.max(y, axis=0)

    tl.store(out_ptr + pid_n * S + s_offs, out, mask=s_mask)


def fused_post_conv(x, multiplier, clamp_min, clamp_max, eps=1e-5):
    # x: [N, C, D, H, W]
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N, C, S)
    mult_flat = multiplier.contiguous().view(C)

    mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
    invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

    stats_kernel[(N * C,)](
        x_flat, mult_flat, mean, invstd,
        N, C, S, eps,
    )

    out = torch.empty((N, S), device=x.device, dtype=torch.float32)
    # next power of 2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = lambda meta: (N, (S + meta['BLOCK_S'] - 1) // meta['BLOCK_S'])
    fused_norm_clamp_max_kernel[grid](
        x_flat, mult_flat, mean, invstd, out,
        N, C, S, clamp_min, clamp_max,
        BLOCK_C=BLOCK_C,
    )

    return out.view(N, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        return fused_post_conv(x, self.multiplier, self.clamp_min, self.clamp_max, eps=1e-5)