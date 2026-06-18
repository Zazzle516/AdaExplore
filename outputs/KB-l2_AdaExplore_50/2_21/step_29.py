import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_bias_scale_sigmoid_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, s_ptr, out_ptr,
    N, IC, IH, IW, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr, OC_C: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # one program per (n, spatial_tile) — produces all OC for that tile
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    sp_off = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    sp_mask = sp_off < (OH * OW)
    oh = sp_off // OW
    ow = sp_off % OW

    # accumulator [BLOCK_SP, OC_C]
    acc = tl.zeros((BLOCK_SP, OC_C), dtype=tl.float32)

    oc_idx = tl.arange(0, OC_C)  # [OC_C]

    x_base = pid_n * IC * IH * IW

    # iterate over IC * KH * KW
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh  # [BLOCK_SP]
                iw = ow + kw  # [BLOCK_SP]
                x_off = x_base + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)  # [BLOCK_SP]
                # weight: [OC, IC, KH, KW] -> [oc, ic, kh, kw]
                w_off = oc_idx * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off)  # [OC_C]
                acc += x_val[:, None] * w_val[None, :]

    # add conv bias + bias + scale + sigmoid
    cb = tl.load(cb_ptr + oc_idx)  # [OC_C]
    b = tl.load(b_ptr + oc_idx)
    s = tl.load(s_ptr + oc_idx)
    y = acc + cb[None, :] + b[None, :]
    y = y * s[None, :]
    y = tl.sigmoid(y)

    # store to output [N, OC, OH, OW]
    out_off = pid_n * OC * OH * OW + oc_idx[None, :] * (OH * OW) + sp_off[:, None]
    tl.store(out_ptr + out_off, y, mask=sp_mask[:, None])


@triton.jit
def groupnorm_kernel(
    x_ptr, out_ptr, gw_ptr, gb_ptr,
    N, C, SPATIAL,
    GROUPS: tl.constexpr, CH_PER_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps,
    BLOCK_SP: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    base = n * C * SPATIAL + g * CH_PER_GROUP * SPATIAL

    sum_val = 0.0
    sum_sq = 0.0

    for ci in tl.static_range(0, CH_PER_GROUP):
        ch_base = base + ci * SPATIAL
        for off in range(0, SPATIAL, BLOCK_SP):
            idx = off + tl.arange(0, BLOCK_SP)
            mask = idx < SPATIAL
            x = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            x = tl.where(mask, x, 0.0)
            sum_val += tl.sum(x)
            sum_sq += tl.sum(x * x)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for ci in tl.static_range(0, CH_PER_GROUP):
        c = g * CH_PER_GROUP + ci
        gw = tl.load(gw_ptr + c)
        gb = tl.load(gb_ptr + c)
        ch_base = base + ci * SPATIAL
        for off in range(0, SPATIAL, BLOCK_SP):
            idx = off + tl.arange(0, BLOCK_SP)
            mask = idx < SPATIAL
            x = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            y = (x - mean) * rstd
            y = y * gw + gb
            tl.store(out_ptr + ch_base + idx, y, mask=mask)


def fused_conv_bias_scale_sigmoid(x, weight, conv_bias, bias, scale):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_SP = 128
    grid = (N, triton.cdiv(OH * OW, BLOCK_SP))
    conv_bias_scale_sigmoid_kernel[grid](
        x, weight, conv_bias, bias, scale, out,
        N, IC, IH, IW, OC, OH, OW,
        KH=KH, KW=KW,
        IC_C=IC, OC_C=OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


def group_norm_only(x, gn_w, gn_b, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    SPATIAL = H * W
    CH_PER_GROUP = C // num_groups
    GROUP_SIZE = CH_PER_GROUP * SPATIAL
    out = torch.empty_like(x)
    grid = (N * num_groups,)
    groupnorm_kernel[grid](
        x, out, gn_w, gn_b,
        N, C, SPATIAL,
        GROUPS=num_groups, CH_PER_GROUP=CH_PER_GROUP,
        GROUP_SIZE=GROUP_SIZE,
        eps=eps,
        BLOCK_SP=1024,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        b = self.bias.view(-1).contiguous()
        s = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        y = fused_conv_bias_scale_sigmoid(x, w, cb, b, s)
        out = group_norm_only(y, gn_w, gn_b, self.num_groups, eps)
        return out