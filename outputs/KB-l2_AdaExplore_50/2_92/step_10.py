import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gn_tanh_hs_res_lse_kernel(
    conv_ptr,        # [N, C, S]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [N, 1, S] -> [N, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    # one program per (n, group, s_block)
    pid_ns = tl.program_id(0)   # n * num_s_blocks + s_block
    pid_g = tl.program_id(1)    # group
    num_s_blocks = tl.cdiv(S, BLOCK_S)
    n = pid_ns // num_s_blocks
    sb = pid_ns % num_s_blocks

    s_offs = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # compute group mean/var across all CH_PER_GROUP channels and all S spatial
    # Stats need full S. We'll compute stats with a separate reduction step.
    # Instead, do stats in a separate kernel per (n, group). But to keep one kernel,
    # we recompute by looping over s tiles for stats.

    # Stats pass:
    sum_val = tl.zeros([CH_PER_GROUP], dtype=tl.float32)
    sumsq_val = tl.zeros([CH_PER_GROUP], dtype=tl.float32)

    c_offs = pid_g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)  # [CH_PER_GROUP]

    for s_start in range(0, S, BLOCK_S):
        s_o = s_start + tl.arange(0, BLOCK_S)
        s_m = s_o < S
        # ptrs: [CH_PER_GROUP, BLOCK_S]
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

    # Now process this s-block: compute output for this n, all channels, this s-block
    # We need full channel reduction for logsumexp. So load conv for all C channels,
    # but stats only available for this group. We need stats for ALL groups for this n.
    # Easier: split into two kernels. But we can do: store mean/rstd implicitly...
    # 
    # Alternative approach: do this kernel only for stats computation per group is fine,
    # but logsumexp needs all channels. Let's restructure: this kernel computes stats and
    # writes them, then a second kernel does the rest.

    # Write stats: one program per (n, group) writes mean/rstd. But we have multiple s-blocks.
    # Only sb==0 writes.
    if sb == 0:
        # store mean and rstd for this (n, g)
        # out_ptr here is repurposed... let's not. We'll use a different approach.
        pass


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

    # we need to compute, for each s in this block:
    #   for c in 0..C-1:
    #     v = conv[n, c, s]
    #     g = c // CH_PER_GROUP
    #     normed = (v - mean[n,g]) * rstd[n,g] * gamma[c] + beta[c]
    #     t = tanh(normed)
    #     hs = t * relu6(t+3)/6
    #     r = v + hs
    #   logsumexp over c
    #
    # Use online logsumexp: keep running max m and sum se = sum(exp(x - m))
    # For each c chunk of BLOCK_C channels.

    m_run = tl.full([BLOCK_S], -float('inf'), dtype=tl.float32)
    se_run = tl.zeros([BLOCK_S], dtype=tl.float32)

    # load all means/rstds for this n: [GROUPS]
    # we'll load them per c chunk

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        # group index
        g_idx = c_offs // CH_PER_GROUP

        # load mean/rstd: [BLOCK_C]
        mean_v = tl.load(mean_ptr + n * GROUPS + g_idx, mask=c_mask, other=0.0)
        rstd_v = tl.load(rstd_ptr + n * GROUPS + g_idx, mask=c_mask, other=0.0)
        gamma_v = tl.load(gamma_ptr + c_offs, mask=c_mask, other=0.0)
        beta_v = tl.load(beta_ptr + c_offs, mask=c_mask, other=0.0)

        # load conv: [BLOCK_C, BLOCK_S]
        ptrs = conv_ptr + n * (C * S) + c_offs[:, None] * S + s_offs[None, :]
        load_mask = c_mask[:, None] & s_mask[None, :]
        v = tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)

        normed = (v - mean_v[:, None]) * rstd_v[:, None] * gamma_v[:, None] + beta_v[:, None]
        # tanh
        t = (tl.exp(2.0 * normed) - 1.0) / (tl.exp(2.0 * normed) + 1.0)
        # hardswish: t * clamp(t+3, 0, 6) / 6
        tp3 = t + 3.0
        clamped = tl.minimum(tl.maximum(tp3, 0.0), 6.0)
        hs = t * clamped / 6.0
        r = v + hs

        # mask out invalid c -> set to -inf
        r = tl.where(c_mask[:, None], r, -float('inf'))

        # online logsumexp update
        # block max along c axis (axis=0)
        block_max = tl.max(r, axis=0)  # [BLOCK_S]
        new_max = tl.maximum(m_run, block_max)
        # update se: se_run * exp(m_run - new_max) + sum(exp(r - new_max))
        # handle -inf safely
        scale = tl.exp(m_run - new_max)
        scale = tl.where(new_max == -float('inf'), 0.0, scale)
        block_sum = tl.sum(tl.exp(r - new_max[None, :]), axis=0)
        se_run = se_run * scale + block_sum
        m_run = new_max

    out = m_run + tl.log(se_run)
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
        BLOCK_S = 256

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

        BLOCK_S2 = 128
        BLOCK_C = min(64, triton.next_power_of_2(C))
        if BLOCK_C < C:
            # ensure BLOCK_C is power of 2 and at least covers reasonably
            BLOCK_C = triton.next_power_of_2(C)

        grid = (N, triton.cdiv(S, BLOCK_S2))
        fused_post_kernel[grid](
            x_flat, mean, rstd, gamma, beta, out_flat,
            N, C, S,
            GROUPS=self.groups,
            CH_PER_GROUP=CH_PER_GROUP,
            BLOCK_S=BLOCK_S2,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )

        return out