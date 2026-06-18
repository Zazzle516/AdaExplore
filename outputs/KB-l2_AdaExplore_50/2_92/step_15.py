import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=16, num_stages=2),
    ],
    key=['C', 'S'],
)
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

    sv = tl.sum(sum_val, axis=0)
    ssq = tl.sum(sum_sq, axis=0)

    total = CHANNELS_PER_GROUP * S
    mean = sv / total
    var = ssq / total - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * GROUPS + pid_g, mean)
    tl.store(invstd_ptr + pid_n * GROUPS + pid_g, inv_std)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=16, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_lse_kernel(
    conv_ptr,
    ab_ptr,        # [N, C, 2] packed (a, b)
    out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    max_val = tl.full((BLOCK_S,), -1e30, tl.float32)
    sum_exp = tl.zeros((BLOCK_S,), tl.float32)

    n_base = pid_n * C * S
    ab_base = pid_n * C * 2

    for c in range(0, C):
        a = tl.load(ab_ptr + ab_base + c * 2)
        b = tl.load(ab_ptr + ab_base + c * 2 + 1)

        x = tl.load(conv_ptr + n_base + c * S + offs_s, mask=mask_s, other=0.0).to(tl.float32)
        norm = x * a + b
        # tanh
        e2 = tl.exp(2.0 * norm)
        t = (e2 - 1.0) / (e2 + 1.0)
        # hardswish: t * relu6(t+3)/6
        relu6_v = tl.minimum(tl.maximum(t + 3.0, 0.0), 6.0)
        hs = t * relu6_v * (1.0 / 6.0)
        val = x + hs

        new_max = tl.maximum(max_val, val)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(val - new_max)
        max_val = new_max

    lse = max_val + tl.log(sum_exp)
    out_base = pid_n * S + offs_s
    tl.store(out_ptr + out_base, lse, mask=mask_s)


@triton.jit
def compute_ab_kernel(
    mean_ptr,    # [N, G]
    invstd_ptr,  # [N, G]
    gamma_ptr,   # [C]
    beta_ptr,    # [C]
    ab_ptr,      # [N, C, 2]
    N, C,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    g = offs_c // CHANNELS_PER_GROUP
    mean = tl.load(mean_ptr + pid_n * GROUPS + g, mask=mask_c, other=0.0)
    inv_std = tl.load(invstd_ptr + pid_n * GROUPS + g, mask=mask_c, other=0.0)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)

    a = inv_std * gamma
    b = beta - mean * a

    ab_off = pid_n * C * 2 + offs_c * 2
    tl.store(ab_ptr + ab_off, a, mask=mask_c)
    tl.store(ab_ptr + ab_off + 1, b, mask=mask_c)


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

        x_flat = x_conv.contiguous().view(N, C, S)

        mean = torch.empty((N, G), device=x_conv.device, dtype=torch.float32)
        invstd = torch.empty((N, G), device=x_conv.device, dtype=torch.float32)

        groupnorm_stats_kernel[(N, G)](
            x_flat, mean, invstd,
            N, C, S,
            GROUPS=G,
            CHANNELS_PER_GROUP=CPG,
            eps=self.eps,
        )

        ab = torch.empty((N, C, 2), device=x_conv.device, dtype=torch.float32)
        BLOCK_C = triton.next_power_of_2(C)
        compute_ab_kernel[(N,)](
            mean, invstd,
            self.group_norm.weight, self.group_norm.bias,
            ab,
            N, C,
            GROUPS=G,
            CHANNELS_PER_GROUP=CPG,
            BLOCK_C=BLOCK_C,
        )

        out = torch.empty((N, 1, H, W), device=x_conv.device, dtype=x_conv.dtype)
        out_flat = out.view(N, S)

        grid = lambda META: (N, triton.cdiv(S, META['BLOCK_S']))

        fused_lse_kernel[grid](
            x_flat, ab, out_flat,
            N, C, S,
        )

        return out