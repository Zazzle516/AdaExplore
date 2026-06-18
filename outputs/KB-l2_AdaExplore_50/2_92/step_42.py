import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gn_tanh_hs_res_lse_kernel(
    conv_ptr,      # [N, C, S]
    gamma_ptr,     # [C]
    beta_ptr,      # [C]
    out_ptr,       # [N, 1, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps,
):
    # one program per (batch, group, spatial-tile)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = pid_g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)  # [CH_PER_GROUP]

    # base pointer to (n, g_start_channel, 0)
    base = pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    mask2d = s_mask[None, :]  # channels are always valid since CH_PER_GROUP divides C

    x = tl.load(conv_ptr + base, mask=mask2d, other=0.0).to(tl.float32)

    # mean and variance over CH_PER_GROUP * S elements (but only valid s)
    # we need to compute over the full group's spatial extent, but each program only sees a tile.
    # So we need to handle this differently: each program needs full S range.
    # Re-design: each program handles one (n, g) entirely, loops over S.

    # NOTE: this kernel assumes BLOCK_S >= S OR we restructure. Will redesign below.


@triton.jit
def gn_stats_kernel(
    conv_ptr,    # [N, C, S]
    mean_ptr,    # [N, GROUPS]
    invstd_ptr,  # [N, GROUPS]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    c_offs = pid_g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)  # [CH_PER_GROUP]

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    num_tiles = tl.cdiv(S, BLOCK_S)
    for t in range(0, num_tiles):
        s_offs = t * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        ptrs = pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
        x = tl.load(conv_ptr + ptrs, mask=s_mask[None, :], other=0.0).to(tl.float32)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    total = CH_PER_GROUP * S
    mean = sum_val / total
    var = sum_sq / total - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * GROUPS + pid_g, mean)
    tl.store(invstd_ptr + pid_n * GROUPS + pid_g, invstd)


@triton.jit
def gn_apply_lse_kernel(
    conv_ptr,    # [N, C, S]
    mean_ptr,    # [N, GROUPS]
    invstd_ptr,  # [N, GROUPS]
    gamma_ptr,   # [C]
    beta_ptr,    # [C]
    out_ptr,     # [N, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # we need to compute, for each spatial position:
    #   max over c in [0,C) of res[c]
    #   sum exp(res[c] - max)
    #   log(sum) + max
    # where res[c] = x[c] + hardswish(tanh(gn[c]))

    # First pass: max
    max_val = tl.full((BLOCK_S,), -float('inf'), dtype=tl.float32)
    for c in range(0, C):
        g = c // CH_PER_GROUP
        mean = tl.load(mean_ptr + pid_n * GROUPS + g)
        invstd = tl.load(invstd_ptr + pid_n * GROUPS + g)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        ptrs = pid_n * C * S + c * S + s_offs
        x = tl.load(conv_ptr + ptrs, mask=s_mask, other=0.0).to(tl.float32)
        norm = (x - mean) * invstd * gamma + beta
        t = tl.extra.cuda.libdevice.tanh(norm)
        # hardswish(t) = t * relu6(t+3) / 6
        hs_in = t + 3.0
        hs_clamped = tl.minimum(tl.maximum(hs_in, 0.0), 6.0)
        hs = t * hs_clamped / 6.0
        res = x + hs
        max_val = tl.maximum(max_val, res)

    # Second pass: sum exp
    sum_exp = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for c in range(0, C):
        g = c // CH_PER_GROUP
        mean = tl.load(mean_ptr + pid_n * GROUPS + g)
        invstd = tl.load(invstd_ptr + pid_n * GROUPS + g)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        ptrs = pid_n * C * S + c * S + s_offs
        x = tl.load(conv_ptr + ptrs, mask=s_mask, other=0.0).to(tl.float32)
        norm = (x - mean) * invstd * gamma + beta
        t = tl.extra.cuda.libdevice.tanh(norm)
        hs_in = t + 3.0
        hs_clamped = tl.minimum(tl.maximum(hs_in, 0.0), 6.0)
        hs = t * hs_clamped / 6.0
        res = x + hs
        sum_exp += tl.exp(res - max_val)

    out = tl.log(sum_exp) + max_val
    out_ptrs = pid_n * S + s_offs
    tl.store(out_ptr + out_ptrs, out, mask=s_mask)


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
        N, C, H, W = x_conv.shape
        S = H * W
        x_flat = x_conv.reshape(N, C, S).contiguous()

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        ch_per_group = C // self.groups

        mean = torch.empty((N, self.groups), device=x_conv.device, dtype=torch.float32)
        invstd = torch.empty((N, self.groups), device=x_conv.device, dtype=torch.float32)

        BLOCK_S_STATS = 1024
        gn_stats_kernel[(N, self.groups)](
            x_flat, mean, invstd,
            N, C, S,
            GROUPS=self.groups,
            CH_PER_GROUP=ch_per_group,
            BLOCK_S=BLOCK_S_STATS,
            eps=float(self.eps),
            num_warps=4,
        )

        out = torch.empty((N, S), device=x_conv.device, dtype=x_conv.dtype)
        BLOCK_S = 256
        grid = (N, triton.cdiv(S, BLOCK_S))
        gn_apply_lse_kernel[grid](
            x_flat, mean, invstd, gamma, beta, out,
            N, C, S,
            GROUPS=self.groups,
            CH_PER_GROUP=ch_per_group,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        return out.view(N, 1, H, W)