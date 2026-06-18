import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def compute_group_stats_kernel(
    conv_ptr,
    mean_ptr,
    invstd_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    CPG: tl.constexpr,
    eps,
    BLOCK_S: tl.constexpr,
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
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def fused_lse_kernel(
    conv_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    max_val = tl.full([BLOCK_S], -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_S], dtype=tl.float32)

    base_ns = pid_n * C * S + s_offs
    base_mg = pid_n * GROUPS

    for c in tl.static_range(0, C):
        g = c // CPG
        mean = tl.load(mean_ptr + base_mg + g)
        invstd = tl.load(invstd_ptr + base_mg + g)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        x = tl.load(conv_ptr + base_ns + c * S, mask=s_mask, other=0.0).to(tl.float32)
        norm = (x - mean) * invstd * gamma + beta
        e2 = tl.exp(2.0 * norm)
        t = (e2 - 1.0) / (e2 + 1.0)
        tp3 = t + 3.0
        tp3_clamped = tl.minimum(tl.maximum(tp3, 0.0), 6.0)
        hs = t * tp3_clamped * (1.0 / 6.0)
        res = x + hs

        new_max = tl.maximum(max_val, res)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(res - new_max)
        max_val = new_max

    lse = tl.log(sum_exp) + max_val
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
        x_conv = self.conv(x)
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