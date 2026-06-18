import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gn_tanh_hs_res_lse_kernel(
    conv_ptr,        # [N, C, H, W]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [N, 1, H, W]
    N, C, H, W,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    HW: tl.constexpr,
    eps,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    # Compute mean and var over C_PER_GROUP * HW elements
    group_size = C_PER_GROUP * HW
    base = n * C * HW + g * C_PER_GROUP * HW

    # Welford-ish: two-pass via simple sum and sum of squares
    sum_val = tl.zeros([BLOCK_HW], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c_inner in tl.static_range(0, C_PER_GROUP):
        c_offset = base + c_inner * HW
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            x = tl.load(conv_ptr + c_offset + offs, mask=mask, other=0.0)
            sum_val += tl.where(mask, x, 0.0)
            sumsq_val += tl.where(mask, x * x, 0.0)

    total_sum = tl.sum(sum_val, axis=0)
    total_sumsq = tl.sum(sumsq_val, axis=0)
    mean = total_sum / group_size
    var = total_sumsq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Now compute LogSumExp per spatial location across all C
    # We need to do this in two passes per spatial tile across the C dimension.
    # But this kernel only handles one group. So we need a different approach:
    # We'll do this in a separate kernel.
    # Instead, write normalized+activated+residual values back, then do LSE separately.

    # Write back: out = x_conv + hardswish(tanh(norm))
    # We'll store into a scratch buffer (using out_ptr is wrong since shape differs)
    # So use a separate output buffer passed differently.
    pass


@triton.jit
def gn_stats_kernel(
    conv_ptr,
    mean_ptr,
    rstd_ptr,
    N, C, H, W,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    HW: tl.constexpr,
    eps,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C_PER_GROUP * HW
    base = n * C * H * W + g * C_PER_GROUP * H * W

    sum_acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
    sumsq_acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c_inner in tl.static_range(0, C_PER_GROUP):
        c_offset = base + c_inner * H * W
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            x = tl.load(conv_ptr + c_offset + offs, mask=mask, other=0.0)
            sum_acc += tl.where(mask, x, 0.0)
            sumsq_acc += tl.where(mask, x * x, 0.0)

    total_sum = tl.sum(sum_acc, axis=0)
    total_sumsq = tl.sum(sumsq_acc, axis=0)
    mean = total_sum / group_size
    var = total_sumsq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * GROUPS + g, mean)
    tl.store(rstd_ptr + n * GROUPS + g, rstd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def fused_apply_lse_kernel(
    conv_ptr,        # [N, C, H, W]
    mean_ptr,        # [N, GROUPS]
    rstd_ptr,        # [N, GROUPS]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [N, 1, H, W]
    N, C, H, W,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    C_CONST: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    HW = H * W
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    n_offset = pid_n * C_CONST * HW
    mean_base = pid_n * GROUPS

    # Online softmax: single pass over channels
    m = tl.full([BLOCK_HW], -float('inf'), dtype=tl.float32)
    s = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c in range(0, C_CONST):
        mean = tl.load(mean_ptr + mean_base + (c // C_PER_GROUP))
        rstd = tl.load(rstd_ptr + mean_base + (c // C_PER_GROUP))
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        # Precompute scale/shift so norm = x_conv * scale + shift
        scale = rstd * gamma
        shift = beta - mean * scale

        x_conv = tl.load(conv_ptr + n_offset + c * HW + hw_offs, mask=hw_mask, other=0.0)
        x_norm = x_conv * scale + shift
        e2x = tl.exp(2.0 * x_norm)
        t = (e2x - 1.0) / (e2x + 1.0)
        tp3 = t + 3.0
        relu6 = tl.minimum(tl.maximum(tp3, 0.0), 6.0)
        hs = t * relu6 * (1.0 / 6.0)
        x_res = x_conv + hs

        m_new = tl.maximum(m, x_res)
        s = s * tl.exp(m - m_new) + tl.exp(x_res - m_new)
        m = m_new

    lse = m + tl.log(s)
    tl.store(out_ptr + pid_n * HW + hw_offs, lse, mask=hw_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        self.groups = groups
        self.out_channels = out_channels
        self.eps = eps

    def forward(self, x):
        x_conv = self.conv(x)
        x_conv = x_conv.contiguous()
        N, C, H, W = x_conv.shape
        HW = H * W
        C_PER_GROUP = C // self.groups

        mean = torch.empty((N, self.groups), device=x_conv.device, dtype=torch.float32)
        rstd = torch.empty((N, self.groups), device=x_conv.device, dtype=torch.float32)

        # Choose BLOCK_HW for stats
        BLOCK_HW_STATS = 2048
        grid_stats = (N * self.groups,)
        gn_stats_kernel[grid_stats](
            x_conv, mean, rstd,
            N, C, H, W,
            GROUPS=self.groups,
            C_PER_GROUP=C_PER_GROUP,
            HW=HW,
            eps=self.eps,
            BLOCK_HW=BLOCK_HW_STATS,
            num_warps=8,
            num_stages=2,
        )

        out = torch.empty((N, 1, H, W), device=x_conv.device, dtype=x_conv.dtype)

        grid = lambda META: (N, triton.cdiv(HW, META['BLOCK_HW']))
        fused_apply_lse_kernel[grid](
            x_conv, mean, rstd,
            self.group_norm.weight, self.group_norm.bias,
            out,
            N, C, H, W,
            GROUPS=self.groups,
            C_PER_GROUP=C_PER_GROUP,
            C_CONST=C,
        )
        return out