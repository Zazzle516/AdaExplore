import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gn_tanh_hs_res_lse_kernel(
    conv_ptr,       # [N, C, S]
    gamma_ptr,      # [C]
    beta_ptr,       # [C]
    out_ptr,        # [N, 1, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CPG: tl.constexpr,    # channels per group
    BLOCK_S: tl.constexpr,
    eps,
):
    # one program per (n, group, s_tile)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    pid_s = tl.program_id(2)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # compute mean & var across channels in group (CPG) and over spatial S
    # We use a two-pass approach: pass over channels in this group for each spatial tile
    # But mean/var must span ALL spatial dims, not just this tile.
    # So we need to loop over all S to compute mean and var.

    # Pass 1: compute sum and sum of squares over full spatial extent for this (n, group)
    sum_val = tl.zeros([], dtype=tl.float32)
    sumsq_val = tl.zeros([], dtype=tl.float32)

    NUM_TILES = (S + BLOCK_S - 1) // BLOCK_S
    for t in range(NUM_TILES):
        cur_s = t * BLOCK_S + tl.arange(0, BLOCK_S)
        cur_mask = cur_s < S
        for ci in range(CPG):
            c = pid_g * CPG + ci
            ptr = conv_ptr + pid_n * C * S + c * S + cur_s
            v = tl.load(ptr, mask=cur_mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(v, axis=0)
            sumsq_val += tl.sum(v * v, axis=0)

    count = CPG * S
    mean = sum_val / count
    var = sumsq_val / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Pass 2: for our s_offs tile, compute logsumexp over C of (x_conv + hardswish(tanh(gn(x_conv))))
    # First compute max over channels for numerical stability
    max_val = tl.full([BLOCK_S], -float('inf'), dtype=tl.float32)

    # We need to iterate over all channels (not just this group). LSE is across all C.
    # Wait: LSE is over dim=1 (all channels), but GN normalization differs per group.
    # So we need stats for ALL groups, not just this one. Let me rethink.

    # Reconsider: One program per (n, s_tile), compute stats for all groups then LSE.
    # But this kernel is per group too. Let me restructure.
    pass


@triton.jit
def compute_group_stats_kernel(
    conv_ptr,    # [N, C, S]
    mean_ptr,    # [N, GROUPS]
    invstd_ptr,  # [N, GROUPS]
    N, C, S,
    GROUPS: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    sum_val = tl.zeros([], dtype=tl.float32)
    sumsq_val = tl.zeros([], dtype=tl.float32)

    NUM_TILES = (S + BLOCK_S - 1) // BLOCK_S
    for ci in range(CPG):
        c = pid_g * CPG + ci
        base = conv_ptr + pid_n * C * S + c * S
        for t in range(NUM_TILES):
            cur_s = t * BLOCK_S + tl.arange(0, BLOCK_S)
            cur_mask = cur_s < S
            v = tl.load(base + cur_s, mask=cur_mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(v, axis=0)
            sumsq_val += tl.sum(v * v, axis=0)

    count = CPG * S
    mean = sum_val / count
    var = sumsq_val / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * GROUPS + pid_g, mean)
    tl.store(invstd_ptr + pid_n * GROUPS + pid_g, inv_std)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_lse_kernel(
    conv_ptr,     # [N, C, S]
    mean_ptr,     # [N, GROUPS]
    invstd_ptr,   # [N, GROUPS]
    gamma_ptr,    # [C]
    beta_ptr,     # [C]
    out_ptr,      # [N, 1, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # Online single-pass LSE
    m = tl.full([BLOCK_S], -float('inf'), dtype=tl.float32)
    s_acc = tl.zeros([BLOCK_S], dtype=tl.float32)

    for c in range(C):
        g = c // CPG
        mean = tl.load(mean_ptr + pid_n * GROUPS + g)
        invstd = tl.load(invstd_ptr + pid_n * GROUPS + g)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        x = tl.load(conv_ptr + pid_n * C * S + c * S + s_offs, mask=s_mask, other=0.0).to(tl.float32)
        norm = (x - mean) * invstd * gamma + beta
        # tanh
        e_pos = tl.exp(norm)
        e_neg = tl.exp(-norm)
        t = (e_pos - e_neg) / (e_pos + e_neg)
        # hardswish(t) = t * relu6(t+3) / 6
        tp3 = t + 3.0
        tp3_clamped = tl.minimum(tl.maximum(tp3, 0.0), 6.0)
        hs = t * tp3_clamped / 6.0
        res = x + hs

        new_m = tl.maximum(m, res)
        s_acc = s_acc * tl.exp(m - new_m) + tl.exp(res - new_m)
        m = new_m

    lse = tl.log(s_acc) + m
    tl.store(out_ptr + pid_n * S + s_offs, lse, mask=s_mask)


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
        self.cpg = out_channels // groups

    def forward(self, x):
        x_conv = self.conv(x)  # [N, C, H, W]
        N, C, H, W = x_conv.shape
        S = H * W
        x_conv_flat = x_conv.reshape(N, C, S).contiguous()

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        mean = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)

        compute_group_stats_kernel[(N, self.groups)](
            x_conv_flat, mean, invstd,
            N, C, S,
            GROUPS=self.groups,
            CPG=self.cpg,
            eps=self.eps,
        )

        out = torch.empty((N, 1, H, W), device=x.device, dtype=x_conv.dtype)
        out_flat = out.reshape(N, S)

        grid = lambda META: (N, (S + META['BLOCK_S'] - 1) // META['BLOCK_S'])
        fused_lse_kernel[grid](
            x_conv_flat, mean, invstd, gamma, beta, out_flat,
            N, C, S,
            GROUPS=self.groups,
            CPG=self.cpg,
        )

        return out