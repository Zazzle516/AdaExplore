import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_bias_scale_sigmoid_kernel(
    x_ptr,        # (N, IC, IH, IW)
    w_ptr,        # (OC, IC, KH, KW)
    cb_ptr,       # conv bias (OC,)
    bias_ptr,     # (OC,)
    scale_ptr,    # (OC,)
    out_ptr,      # (N, OC, OH, OW)
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, OC // BLOCK_OC, ceil(OH*OW / BLOCK_HW))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]

    oh = hw_offs // OW
    ow = hw_offs % OW
    hw_mask = hw_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Loop over IC, KH, KW
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            ih = oh + kh  # padding=0
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                # load input slice: shape [BLOCK_HW]
                x_off = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_mask = hw_mask
                x = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_HW]
                # load weights for this (ic, kh, kw): shape [BLOCK_OC]
                w_off = oc_offs * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += w[:, None] * x[None, :]

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += cb[:, None]

    # Add extra bias
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    s = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = (acc + b[:, None]) * s[:, None]
    acc = tl.sigmoid(acc)

    # Store: output layout (N, OC, OH*OW)
    out_off = (pid_n * OC + oc_offs[:, None]) * (OH * OW) + hw_offs[None, :]
    mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def gn_kernel(
    x_ptr,          # (N, C, HW)
    gn_w_ptr,       # (C,)
    gn_b_ptr,       # (C,)
    out_ptr,        # (N, C, HW)
    N, C, HW,
    num_groups,
    group_size,
    eps,
    BLOCK_HW: tl.constexpr,
    CPG: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups
    c_start = g * CPG

    sum_x = 0.0
    sum_x2 = 0.0

    base = n * C * HW + c_start * HW

    for ci in tl.static_range(0, CPG):
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            ptr = x_ptr + base + ci * HW + offs
            x = tl.load(ptr, mask=mask, other=0.0)
            sum_x += tl.sum(x, axis=0)
            sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for ci in tl.static_range(0, CPG):
        c = c_start + ci
        gw = tl.load(gn_w_ptr + c)
        gb = tl.load(gn_b_ptr + c)
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            ptr = x_ptr + base + ci * HW + offs
            x = tl.load(ptr, mask=mask, other=0.0)
            y = (x - mean) * rstd * gw + gb
            tl.store(out_ptr + base + ci * HW + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()

        conv_out = torch.empty((N, OC, OH * OW), device=x.device, dtype=x.dtype)

        BLOCK_HW = 128
        BLOCK_OC = 32
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))

        conv_bias_scale_sigmoid_kernel[grid](
            x, w, cb, bias_flat, scale_flat, conv_out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            IC_C=IC,
            BLOCK_HW=BLOCK_HW,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        out = torch.empty_like(conv_out)
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        channels_per_group = OC // self.num_groups
        group_size = channels_per_group * OH * OW

        gn_grid = (N * self.num_groups,)
        gn_kernel[gn_grid](
            conv_out, gn_w, gn_b, out,
            N, OC, OH * OW,
            self.num_groups,
            group_size,
            self.eps,
            BLOCK_HW=1024,
            CPG=channels_per_group,
            num_warps=8,
        )

        return out.view(N, OC, OH, OW)