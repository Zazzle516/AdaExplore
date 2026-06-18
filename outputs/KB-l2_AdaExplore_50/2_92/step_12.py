import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gn_stats_kernel(
    conv_ptr,        # [N, C, S]
    mean_ptr,        # [N, GROUPS]
    rstd_ptr,        # [N, GROUPS]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    c_offs = g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)
    sum_val = tl.zeros([CH_PER_GROUP], dtype=tl.float32)
    sumsq_val = tl.zeros([CH_PER_GROUP], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_o = s_start + tl.arange(0, BLOCK_S)
        s_m = s_o < S
        ptrs = conv_ptr + n * (C * S) + c_offs[:, None] * S + s_o[None, :]
        x = tl.load(ptrs, mask=s_m[None, :], other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=1)
        sumsq_val += tl.sum(x * x, axis=1)

    total_sum = tl.sum(sum_val, axis=0)
    total_sumsq = tl.sum(sumsq_val, axis=0)
    count = CH_PER_GROUP * S
    mean = total_sum / count
    var = total_sumsq / count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * GROUPS + g, mean)
    tl.store(rstd_ptr + n * GROUPS + g, rstd)


@triton.jit
def fused_post_kernel(
    conv_ptr,        # [N, C, S]
    mean_ptr,        # [N, GROUPS]
    rstd_ptr,        # [N, GROUPS]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    sb = tl.program_id(1)

    s_offs = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # Load all gamma/beta/mean/rstd for full C (BLOCK_C == C, power of 2 covering C)
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    g_idx = c_offs // CH_PER_GROUP

    mean_v = tl.load(mean_ptr + n * GROUPS + g_idx, mask=c_mask, other=0.0)
    rstd_v = tl.load(rstd_ptr + n * GROUPS + g_idx, mask=c_mask, other=0.0)
    gamma_v = tl.load(gamma_ptr + c_offs, mask=c_mask, other=0.0)
    beta_v = tl.load(beta_ptr + c_offs, mask=c_mask, other=0.0)

    # Load conv [BLOCK_C, BLOCK_S]
    ptrs = conv_ptr + n * (C * S) + c_offs[:, None] * S + s_offs[None, :]
    load_mask = c_mask[:, None] & s_mask[None, :]
    v = tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)

    normed = (v - mean_v[:, None]) * rstd_v[:, None] * gamma_v[:, None] + beta_v[:, None]
    # tanh via fast formula: 1 - 2/(exp(2x)+1)
    e2 = tl.exp(2.0 * normed)
    t = 1.0 - 2.0 / (e2 + 1.0)
    # hardswish: t * clamp(t+3, 0, 6) / 6
    tp3 = t + 3.0
    clamped = tl.minimum(tl.maximum(tp3, 0.0), 6.0)
    hs = t * clamped * (1.0 / 6.0)
    r = v + hs

    r = tl.where(c_mask[:, None], r, -float('inf'))

    # logsumexp along c (axis=0): single-pass since BLOCK_C covers all C
    m = tl.max(r, axis=0)  # [BLOCK_S]
    se = tl.sum(tl.exp(r - m[None, :]), axis=0)
    out = m + tl.log(se)
    tl.store(out_ptr + n * S + s_offs, out, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.eps = eps
        self.groups = groups
        self.out_channels = out_channels

    def forward(self, x):
        x_conv = self.conv(x)
        N, C, H, W = x_conv.shape
        S = H * W
        x_flat = x_conv.contiguous().view(N, C, S)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        mean = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)

        CH_PER_GROUP = C // self.groups
        BLOCK_S = 512

        gn_stats_kernel[(N, self.groups)](
            x_flat, mean, rstd,
            N, C, S,
            GROUPS=self.groups,
            CH_PER_GROUP=CH_PER_GROUP,
            BLOCK_S=BLOCK_S,
            eps=self.eps,
            num_warps=4,
        )

        out = torch.empty((N, 1, H, W), device=x.device, dtype=torch.float32)
        out_flat = out.view(N, S)

        BLOCK_C = triton.next_power_of_2(C)
        BLOCK_S2 = 128

        grid = (N, triton.cdiv(S, BLOCK_S2))
        fused_post_kernel[grid](
            x_flat, mean, rstd, gamma, beta, out_flat,
            N, C, S,
            GROUPS=self.groups,
            CH_PER_GROUP=CH_PER_GROUP,
            BLOCK_S=BLOCK_S2,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )

        return out