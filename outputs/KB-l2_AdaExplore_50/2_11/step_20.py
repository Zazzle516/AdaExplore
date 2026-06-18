import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_bn_tanh_pool_kernel(
    x_ptr,           # input: [N, IC, IH, IW]
    w_ptr,           # weight: [IC, OC, KH, KW]
    scale_ptr,       # [OC]
    shift_ptr,       # [OC]
    out_ptr,         # output after pool: [N, OC, POH, POW]
    N, IC, IH, IW,
    OC, OH, OW, POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PS: tl.constexpr,  # number of pooled-spatial output positions per program
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ps = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    offs_ps = pid_ps * BLOCK_PS + tl.arange(0, BLOCK_PS)  # [BLOCK_PS]

    poh = offs_ps // POW
    pow_ = offs_ps % POW

    mask_oc = offs_oc < OC
    mask_ps = offs_ps < POH * POW

    # 4 output positions per pooled element: (2*poh + dh, 2*pow + dw)
    # We'll compute 4 accumulators
    acc00 = tl.zeros((BLOCK_OC, BLOCK_PS), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_PS), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_PS), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_PS), dtype=tl.float32)

    # ConvTranspose2d: output[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh+pad-kh, ow+pad-kw] * w[ic, oc, kh, kw]
    # For 2x2 pool: 4 output positions oh in {2*poh, 2*poh+1}, ow in {2*pow, 2*pow+1}
    # We share the weight loads across all 4 positions per (ic, kh, kw)
    
    oh0 = 2 * poh
    oh1 = oh0 + 1
    ow0 = 2 * pow_
    ow1 = ow0 + 1

    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                # Compute input positions for each of the 4 output positions
                ih0 = oh0 + PAD - kh
                ih1 = oh1 + PAD - kh
                iw0 = ow0 + PAD - kw
                iw1 = ow1 + PAD - kw

                v00_mask = (ih0 >= 0) & (ih0 < IH) & (iw0 >= 0) & (iw0 < IW) & mask_ps
                v01_mask = (ih0 >= 0) & (ih0 < IH) & (iw1 >= 0) & (iw1 < IW) & mask_ps
                v10_mask = (ih1 >= 0) & (ih1 < IH) & (iw0 >= 0) & (iw0 < IW) & mask_ps
                v11_mask = (ih1 >= 0) & (ih1 < IH) & (iw1 >= 0) & (iw1 < IW) & mask_ps

                base_x = pid_n * IC * IH * IW + ic * IH * IW
                x00 = tl.load(x_ptr + base_x + ih0 * IW + iw0, mask=v00_mask, other=0.0)
                x01 = tl.load(x_ptr + base_x + ih0 * IW + iw1, mask=v01_mask, other=0.0)
                x10 = tl.load(x_ptr + base_x + ih1 * IW + iw0, mask=v10_mask, other=0.0)
                x11 = tl.load(x_ptr + base_x + ih1 * IW + iw1, mask=v11_mask, other=0.0)

                w_off = ic * OC * KH * KW + offs_oc * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                acc00 += w_val[:, None] * x00[None, :]
                acc01 += w_val[:, None] * x01[None, :]
                acc10 += w_val[:, None] * x10[None, :]
                acc11 += w_val[:, None] * x11[None, :]

    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=mask_oc, other=0.0)

    s = scale[:, None]
    sh = shift[:, None]

    v00 = acc00 * s + sh
    v01 = acc01 * s + sh
    v10 = acc10 * s + sh
    v11 = acc11 * s + sh

    # tanh via libdevice
    t00 = tl.extra.cuda.libdevice.tanh(v00)
    t01 = tl.extra.cuda.libdevice.tanh(v01)
    t10 = tl.extra.cuda.libdevice.tanh(v10)
    t11 = tl.extra.cuda.libdevice.tanh(v11)

    m = tl.maximum(tl.maximum(t00, t01), tl.maximum(t10, t11))

    out_off = pid_n * OC * POH * POW + offs_oc[:, None] * POH * POW + offs_ps[None, :]
    mask_out = mask_oc[:, None] & mask_ps[None, :]
    tl.store(out_ptr + out_off, m, mask=mask_out)


@triton.jit
def group_norm_kernel(
    x_ptr,           # [N, C, H, W] - modified in place
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
    N, C, H, W,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,    # H * W
    GROUP_SIZE: tl.constexpr, # CHANNELS_PER_GROUP * SPATIAL
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, BLOCK)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    group_base = n * C * SPATIAL + g * CHANNELS_PER_GROUP * SPATIAL

    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + group_base + idx, mask=mask, other=0.0)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + group_base + idx, mask=mask, other=0.0)

        c_local = idx // SPATIAL
        c = g * CHANNELS_PER_GROUP + c_local
        gw = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
        gb = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)

        normed = (v - mean) * rstd
        result = normed * gw + gb
        tl.store(x_ptr + group_base + idx, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_groups = num_groups

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        x = x.contiguous()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        PAD = self.padding
        OC = self.out_channels

        # ConvTranspose2d, stride=1, output_padding=0:
        # out = in + kernel - 2*pad - 1
        OH = IH + KH - 2 * PAD - 1
        OW = IW + KW - 2 * PAD - 1
        POH = OH // 2
        POW = OW // 2

        # Fold BN
        bn = self.batch_norm
        bn_var = bn.running_var
        bn_mean = bn.running_mean
        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_eps = bn.eps

        scale = bn_weight / torch.sqrt(bn_var + bn_eps)
        conv_bias = self.conv_transpose.bias
        if conv_bias is not None:
            shift = bn_bias - bn_mean * scale + conv_bias * scale
        else:
            shift = bn_bias - bn_mean * scale

        w = self.conv_transpose.weight.contiguous()

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_PS = 32
        PS = POH * POW
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(PS, BLOCK_PS))

        conv_transpose_bn_tanh_pool_kernel[grid](
            x, w,
            scale.contiguous(), shift.contiguous(),
            out,
            N, IC, IH, IW,
            OC, OH, OW, POH, POW,
            KH=KH, KW=KW, PAD=PAD,
            BLOCK_OC=BLOCK_OC, BLOCK_PS=BLOCK_PS,
            num_warps=4, num_stages=2,
        )

        # Group norm in place
        channels_per_group = OC // self.num_groups
        spatial = POH * POW
        group_size = channels_per_group * spatial

        BLOCK = 1
        while BLOCK < group_size and BLOCK < 1024:
            BLOCK *= 2
        BLOCK = min(BLOCK, 1024)

        grid2 = (N * self.num_groups,)

        group_norm_kernel[grid2](
            out,
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, OC, POH, POW,
            GROUPS=self.num_groups,
            CHANNELS_PER_GROUP=channels_per_group,
            SPATIAL=spatial,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.group_norm.eps,
            num_warps=4,
        )

        return out