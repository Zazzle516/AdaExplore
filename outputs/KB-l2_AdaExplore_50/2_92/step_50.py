import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=3),
    ],
    key=['N', 'OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr,        # [N, IC, H, W]
    w_ptr,        # [OC, IC, KH, KW]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    s_mask = s_offs < (OH * OW)

    oh = s_offs // OW
    ow = s_offs % OW

    oc_offs = tl.arange(0, OC)  # [OC]

    # accumulator [OC, BLOCK_S]
    acc = tl.zeros((OC, BLOCK_S), dtype=tl.float32)

    # Loop over IC, KH, KW
    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh  # [BLOCK_S]
                iw = ow + kw  # [BLOCK_S]
                # input pointer [BLOCK_S]
                x_ptrs = pid_n * IC * H * W + ic * H * W + ih * W + iw
                x_val = tl.load(x_ptr + x_ptrs, mask=s_mask, other=0.0)  # [BLOCK_S]

                # weight pointer [OC]
                w_ptrs = oc_offs * IC * KH * KW + ic * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_ptrs)  # [OC]

                acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs)  # [OC]
    acc += bias[:, None]

    # store [OC, BLOCK_S] -> out[n, oc, s]
    out_ptrs = pid_n * OC * OH * OW + oc_offs[:, None] * (OH * OW) + s_offs[None, :]
    tl.store(out_ptr + out_ptrs, acc, mask=s_mask[None, :])


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

    c_offs = pid_g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)

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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def gn_apply_lse_kernel(
    conv_ptr,    # [N, C, S]
    mean_ptr,    # [N, GROUPS]
    invstd_ptr,  # [N, GROUPS]
    gamma_ptr,   # [C]
    beta_ptr,    # [C]
    out_ptr,     # [N, S]
    N, C: tl.constexpr, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, C)
    g_offs = c_offs // CH_PER_GROUP

    mean = tl.load(mean_ptr + pid_n * GROUPS + g_offs)
    invstd = tl.load(invstd_ptr + pid_n * GROUPS + g_offs)
    gamma = tl.load(gamma_ptr + c_offs)
    beta = tl.load(beta_ptr + c_offs)

    scale = invstd * gamma
    shift = beta - mean * scale

    ptrs = pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    mask2d = s_mask[None, :]
    x = tl.load(conv_ptr + ptrs, mask=mask2d, other=0.0).to(tl.float32)

    norm = x * scale[:, None] + shift[:, None]
    t = tl.extra.cuda.libdevice.tanh(norm)
    hs_in = t + 3.0
    hs_clamped = tl.minimum(tl.maximum(hs_in, 0.0), 6.0)
    hs = t * hs_clamped * (1.0 / 6.0)
    res = x + hs

    max_val = tl.max(res, axis=0)
    sum_exp = tl.sum(tl.exp(res - max_val[None, :]), axis=0)

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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        S = OH * OW

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        x_conv = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        grid_conv = lambda meta: (N, triton.cdiv(S, meta['BLOCK_S']))
        conv2d_kernel[grid_conv](
            x, weight, bias, x_conv,
            N, IC, H, W, OC, OH, OW, KH, KW,
        )

        x_flat = x_conv.view(N, OC, S)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        ch_per_group = OC // self.groups

        mean = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)

        BLOCK_S_STATS = 2048
        gn_stats_kernel[(N, self.groups)](
            x_flat, mean, invstd,
            N, OC, S,
            GROUPS=self.groups,
            CH_PER_GROUP=ch_per_group,
            BLOCK_S=BLOCK_S_STATS,
            eps=float(self.eps),
            num_warps=8,
            num_stages=2,
        )

        out = torch.empty((N, S), device=x.device, dtype=torch.float32)
        grid = lambda meta: (N, triton.cdiv(S, meta['BLOCK_S']))
        gn_apply_lse_kernel[grid](
            x_flat, mean, invstd, gamma, beta, out,
            N, OC, S,
            GROUPS=self.groups,
            CH_PER_GROUP=ch_per_group,
        )

        return out.view(N, 1, OH, OW)