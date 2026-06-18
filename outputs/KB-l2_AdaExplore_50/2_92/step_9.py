import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def groupnorm_stats_kernel(
    conv_ptr,
    mean_ptr,
    invstd_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    base = pid_n * C * S + pid_g * CHANNELS_PER_GROUP * S

    sum_val = tl.zeros((BLOCK_S,), tl.float32)
    sum_sq = tl.zeros((BLOCK_S,), tl.float32)
    offs_s = tl.arange(0, BLOCK_S)

    for c in tl.static_range(0, CHANNELS_PER_GROUP):
        c_off = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + offs_s
            mask = offs < S
            x = tl.load(conv_ptr + c_off + offs, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.where(mask, x, 0.0)
            sum_sq += tl.where(mask, x * x, 0.0)

    s_v = tl.sum(sum_val, axis=0)
    sq_v = tl.sum(sum_sq, axis=0)

    total = CHANNELS_PER_GROUP * S
    mean = s_v / total
    var = sq_v / total - mean * mean
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
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_lse_kernel(
    conv_ptr,        # [N, C, S]
    mean_ptr,        # [N, GROUPS]
    invstd_ptr,      # [N, GROUPS]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    max_val = tl.full((BLOCK_S,), -1e30, tl.float32)
    sum_exp = tl.zeros((BLOCK_S,), tl.float32)

    n_base = pid_n * C * S
    mean_base = pid_n * GROUPS

    for c in range(0, C):
        g = c // CHANNELS_PER_GROUP
        mean = tl.load(mean_ptr + mean_base + g)
        inv_std = tl.load(invstd_ptr + mean_base + g)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)
        a = inv_std * gamma
        b = beta - mean * a

        x = tl.load(conv_ptr + n_base + c * S + offs_s, mask=mask_s, other=0.0).to(tl.float32)
        norm = x * a + b
        # tanh via fast path
        e2 = tl.exp(2.0 * norm)
        t = (e2 - 1.0) / (e2 + 1.0)
        # hardswish on t
        relu6_v = tl.minimum(tl.maximum(t + 3.0, 0.0), 6.0)
        hs = t * relu6_v * (1.0 / 6.0)
        val = x + hs

        new_max = tl.maximum(max_val, val)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(val - new_max)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)
    out_base = pid_n * S + offs_s
    tl.store(out_ptr + out_base, lse, mask=mask_s)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        self.eps = eps
        self.groups = groups
        self.out_channels = out_channels

    def forward(self, x):
        x_conv = self.conv(x)
        N, C, H, W = x_conv.shape
        S = H * W
        G = self.groups
        CPG = C // G

        x_conv_c = x_conv.contiguous()
        x_flat = x_conv_c.view(N, C, S)

        mean = torch.empty((N, G), device=x_conv.device, dtype=torch.float32)
        invstd = torch.empty((N, G), device=x_conv.device, dtype=torch.float32)

        BLOCK_S_STATS = 1024
        if S < 1024:
            BLOCK_S_STATS = triton.next_power_of_2(S)
            if BLOCK_S_STATS < 64:
                BLOCK_S_STATS = 64

        groupnorm_stats_kernel[(N, G)](
            x_flat, mean, invstd,
            N, C, S,
            GROUPS=G,
            CHANNELS_PER_GROUP=CPG,
            BLOCK_S=BLOCK_S_STATS,
            eps=self.eps,
            num_warps=8,
        )

        out = torch.empty((N, 1, H, W), device=x_conv.device, dtype=x_conv.dtype)
        out_flat = out.view(N, S)

        grid = lambda META: (N, triton.cdiv(S, META['BLOCK_S']))

        fused_lse_kernel[grid](
            x_flat,
            mean, invstd,
            self.group_norm.weight, self.group_norm.bias,
            out_flat,
            N, C, S,
            GROUPS=G,
            CHANNELS_PER_GROUP=CPG,
        )

        return out