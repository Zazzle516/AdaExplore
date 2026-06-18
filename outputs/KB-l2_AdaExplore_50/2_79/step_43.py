import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)
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
    c_mask = c_offs < C

    m = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)
    mean = tl.load(mean_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)
    invstd = tl.load(invstd_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)

    # Combine: y = clamp((x*m - mean)*invstd, lo, hi) * m
    # = clamp(x * (m*invstd) - mean*invstd, lo, hi) * m
    a = m * invstd          # [BLOCK_C]
    b = mean * invstd       # [BLOCK_C]

    x_ptrs = x_ptr + pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    full_mask = c_mask[:, None] & s_mask[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=0.0)

    y = x * a[:, None] - b[:, None]
    y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
    y = y * m[:, None]

    neg_inf = float("-inf")
    y_masked = tl.where(c_mask[:, None], y, neg_inf)
    out = tl.max(y_masked, axis=0)

    tl.store(out_ptr + pid_n * S + s_offs, out, mask=s_mask)


def fused_post_conv(x, multiplier, clamp_min, clamp_max, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N, C, S)
    mult_flat = multiplier.contiguous().view(C)

    mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
    invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

    # Choose BLOCK_S for stats based on S
    BLOCK_S_STATS = 1024
    stats_kernel[(N * C,)](
        x_flat, mult_flat, mean, invstd,
        N, C, S, eps,
        BLOCK_S=BLOCK_S_STATS,
        num_warps=4,
    )

    out = torch.empty((N, S), device=x.device, dtype=torch.float32)
    BLOCK_S = 256
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)
    fused_norm_clamp_max_kernel[grid](
        x_flat, mult_flat, mean, invstd, out,
        N, C, S, clamp_min, clamp_max,
        BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C,
        num_warps=4,
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