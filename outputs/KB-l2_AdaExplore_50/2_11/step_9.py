import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_bn_tanh_kernel(
    x_ptr,           # input: [N, IC, IH, IW]
    w_ptr,           # weight: [IC, OC, KH, KW]
    bias_ptr,        # [OC] folded bias
    scale_ptr,       # [OC] folded scale (bn_weight/sqrt(var+eps))
    shift_ptr,       # [OC] folded shift (bn_bias - bn_mean*scale + conv_bias*scale)
    out_ptr,         # output: [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)

    oh = offs_s // OW
    ow = offs_s % OW

    mask_oc = offs_oc < OC
    mask_s = offs_s < OH * OW

    acc = tl.zeros((BLOCK_OC, BLOCK_S), dtype=tl.float32)

    # ConvTranspose2d: output[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih = oh + pad - kh, iw = ow + pad - kw  (stride=1)
    for ic in range(IC):
        for kh in range(KH):
            for kw in range(KW):
                ih = oh + PAD - kh
                iw = ow + PAD - kw
                valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & mask_s

                x_off = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_S]

                w_off = ic * OC * KH * KW + offs_oc * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # apply BN scale/shift, then tanh
    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)
    shift = tl.load(shift_ptr + offs_oc, mask=mask_oc, other=0.0)

    out = acc * scale[:, None] + shift[:, None]
    # tanh
    out = (tl.exp(2.0 * out) - 1.0) / (tl.exp(2.0 * out) + 1.0)

    out_off = pid_n * OC * OH * OW + offs_oc[:, None] * OH * OW + offs_s[None, :]
    mask_out = mask_oc[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_off, out, mask=mask_out)


@triton.jit
def fused_maxpool_gn_kernel(
    x_ptr,           # [N, C, H, W]
    out_ptr,         # [N, C, OH, OW]
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
    N, C, H, W, OH, OW,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, BLOCK)

    sum_val = 0.0
    sum_sq = 0.0

    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE

        c_local = idx // POOL_SIZE
        spatial = idx % POOL_SIZE
        oh = spatial // OW
        ow = spatial % OW

        c = g * CHANNELS_PER_GROUP + c_local

        ih0 = 2 * oh
        iw0 = 2 * ow

        base = n * C * H * W + c * H * W

        p00 = tl.load(x_ptr + base + ih0 * W + iw0, mask=mask, other=0.0)
        p01 = tl.load(x_ptr + base + ih0 * W + iw0 + 1, mask=mask, other=0.0)
        p10 = tl.load(x_ptr + base + (ih0 + 1) * W + iw0, mask=mask, other=0.0)
        p11 = tl.load(x_ptr + base + (ih0 + 1) * W + iw0 + 1, mask=mask, other=0.0)

        m1 = tl.maximum(p00, p01)
        m2 = tl.maximum(p10, p11)
        m = tl.maximum(m1, m2)

        m = tl.where(mask, m, 0.0)

        out_offset = n * C * OH * OW + c * OH * OW + oh * OW + ow
        tl.store(out_ptr + out_offset, m, mask=mask)

        sum_val += tl.sum(m, axis=0)
        sum_sq += tl.sum(m * m, axis=0)

    mean = sum_val / GROUP_SIZE
    var = sum_sq / GROUP_SIZE - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    for start in range(0, GROUP_SIZE, BLOCK):
        idx = start + offs
        mask = idx < GROUP_SIZE

        c_local = idx // POOL_SIZE
        spatial = idx % POOL_SIZE
        oh = spatial // OW
        ow = spatial % OW
        c = g * CHANNELS_PER_GROUP + c_local

        out_offset = n * C * OH * OW + c * OH * OW + oh * OW + ow
        v = tl.load(out_ptr + out_offset, mask=mask, other=0.0)

        gw = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
        gb = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)

        normed = (v - mean) * rstd
        result = normed * gw + gb

        tl.store(out_ptr + out_offset, result, mask=mask)


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
        # ConvTranspose2d output size with stride=1:
        OH = IH + 2 * (KH - 1) - 2 * PAD  # = IH + KH - 1 - 2*PAD ... wait
        # Formula: out = (in - 1) * stride - 2*pad + (kernel - 1) + output_padding + 1
        # stride=1, output_padding=0: out = in - 1 - 2*pad + kernel - 1 + 1 = in + kernel - 2*pad - 1
        OH = IH + KH - 2 * PAD - 1
        OW = IW + KW - 2 * PAD - 1

        # Fold BN into conv epilogue
        bn = self.batch_norm
        bn_var = bn.running_var
        bn_mean = bn.running_mean
        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_eps = bn.eps

        scale = bn_weight / torch.sqrt(bn_var + bn_eps)  # [OC]
        conv_bias = self.conv_transpose.bias  # [OC]
        if conv_bias is not None:
            shift = bn_bias - bn_mean * scale + conv_bias * scale
        else:
            shift = bn_bias - bn_mean * scale

        # Conv weight: [IC, OC, KH, KW]
        w = self.conv_transpose.weight.contiguous()

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_S = 64
        S = OH * OW
        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(S, BLOCK_S))

        conv_transpose_bn_tanh_kernel[grid](
            x, w, self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device),
            scale.contiguous(), shift.contiguous(),
            conv_out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW, PAD=PAD,
            BLOCK_OC=BLOCK_OC, BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2,
        )

        # Now max pool + group norm
        POH = OH // 2
        POW = OW // 2
        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        channels_per_group = OC // self.num_groups
        pool_size = POH * POW
        group_size = channels_per_group * pool_size

        BLOCK = 1
        while BLOCK < group_size and BLOCK < 1024:
            BLOCK *= 2
        BLOCK = min(BLOCK, 1024)

        grid2 = (N * self.num_groups,)

        fused_maxpool_gn_kernel[grid2](
            conv_out, out,
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, OC, OH, OW, POH, POW,
            GROUPS=self.num_groups,
            CHANNELS_PER_GROUP=channels_per_group,
            POOL_SIZE=pool_size,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.group_norm.eps,
            num_warps=4,
        )

        return out