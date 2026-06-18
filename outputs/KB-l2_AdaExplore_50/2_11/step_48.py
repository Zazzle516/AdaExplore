import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_bn_tanh_pool_kernel(
    x_ptr,       # input [N, IC, H, W]
    w_ptr,       # conv_transpose weight reshaped as [OC, IC*KH*KW] (after flip+permute)
    bias_ptr,    # fused bias [OC]
    scale_ptr,   # fused scale [OC]
    out_ptr,     # output after pool [N, OC, H_out, W_out]
    N, IC, H, W,
    OC, H_o, W_o,
    PAD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_oc = tl.program_id(2)

    hw_start = pid_hw * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (H_o * W_o)

    # pooled output coordinates
    oh = offs_hw // W_o
    ow = offs_hw % W_o
    # pre-pool top-left
    oh_p = oh * 2
    ow_p = ow * 2

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = oc_offs < OC

    acc00 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    n_off = pid_n * IC * H * W

    # weight pre-arranged as [OC, IC, KH, KW] (flipped over KH,KW)
    # so weight index = oc*IC*KH*KW + ic*KH*KW + kh*KW + kw
    # For ConvTranspose with stride=1: ih = oh_p + PAD - kh
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih00 = oh_p + PAD - kh
            iw00 = ow_p + PAD - kw
            ih01 = ih00
            iw01 = iw00 + 1
            ih10 = ih00 + 1
            iw10 = iw00
            ih11 = ih00 + 1
            iw11 = iw00 + 1

            v00 = (ih00 >= 0) & (ih00 < H) & (iw00 >= 0) & (iw00 < W)
            v01 = (ih01 >= 0) & (ih01 < H) & (iw01 >= 0) & (iw01 < W)
            v10 = (ih10 >= 0) & (ih10 < H) & (iw10 >= 0) & (iw10 < W)
            v11 = (ih11 >= 0) & (ih11 < H) & (iw11 >= 0) & (iw11 < W)

            for ic in range(0, IC):
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)

                ic_off = n_off + ic * H * W

                x00 = tl.load(x_ptr + ic_off + ih00 * W + iw00, mask=v00 & mask_hw, other=0.0)
                acc00 += w_val[:, None] * x00[None, :]

                x01 = tl.load(x_ptr + ic_off + ih01 * W + iw01, mask=v01 & mask_hw, other=0.0)
                acc01 += w_val[:, None] * x01[None, :]

                x10 = tl.load(x_ptr + ic_off + ih10 * W + iw10, mask=v10 & mask_hw, other=0.0)
                acc10 += w_val[:, None] * x10[None, :]

                x11 = tl.load(x_ptr + ic_off + ih11 * W + iw11, mask=v11 & mask_hw, other=0.0)
                acc11 += w_val[:, None] * x11[None, :]

    scale = tl.load(scale_ptr + oc_offs, mask=mask_oc, other=0.0)
    shift = tl.load(bias_ptr + oc_offs, mask=mask_oc, other=0.0)

    sc = scale[:, None]
    sh = shift[:, None]

    v00f = acc00 * sc + sh
    v01f = acc01 * sc + sh
    v10f = acc10 * sc + sh
    v11f = acc11 * sc + sh

    t00 = 2.0 * tl.sigmoid(2.0 * v00f) - 1.0
    t01 = 2.0 * tl.sigmoid(2.0 * v01f) - 1.0
    t10 = 2.0 * tl.sigmoid(2.0 * v10f) - 1.0
    t11 = 2.0 * tl.sigmoid(2.0 * v11f) - 1.0

    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)

    out_off = pid_n * OC * H_o * W_o + oc_offs[:, None] * (H_o * W_o) + offs_hw[None, :]
    store_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, pooled, mask=store_mask)


@triton.jit
def group_norm_kernel(
    x_ptr,
    out_ptr,
    weight_ptr,
    bias_ptr,
    N, C, HW,
    G,
    eps,
    CPG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    c_base = g * CPG
    total = CPG * HW

    offs_c = tl.arange(0, CPG)
    offs_s = tl.arange(0, BLOCK)
    mask_s = offs_s < HW

    n_off = n * C * HW
    c_off = (c_base + offs_c)[:, None] * HW
    s_off = offs_s[None, :]
    mask2 = mask_s[None, :]

    x = tl.load(x_ptr + n_off + c_off + s_off, mask=mask2, other=0.0)
    x_m = tl.where(mask2, x, 0.0)
    s = tl.sum(x_m)
    sq = tl.sum(x_m * x_m)
    mean = s / total
    var = sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    gw = tl.load(weight_ptr + c_base + offs_c)
    gb = tl.load(bias_ptr + c_base + offs_c)

    y = (x - mean) * rstd * gw[:, None] + gb[:, None]
    tl.store(out_ptr + n_off + c_off + s_off, y, mask=mask2)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self._cached_weight = None

    def _get_fused_weight(self):
        # conv_transpose.weight shape: [IC, OC, KH, KW]
        # We want [OC, IC, KH, KW] with KH,KW flipped
        w = self.conv_transpose.weight  # [IC, OC, KH, KW]
        w = w.permute(1, 0, 2, 3).contiguous()  # [OC, IC, KH, KW]
        w = torch.flip(w, dims=[2, 3]).contiguous()
        return w

    def forward(self, x):
        if self.training or self.stride != 1:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        bn = self.batch_norm
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bn_shift = bn.bias - bn.running_mean * bn_scale
        fused_shift = (bn_scale * self.conv_transpose.bias + bn_shift).contiguous()
        fused_scale = bn_scale.contiguous()

        if self._cached_weight is None or self._cached_weight.device != x.device:
            self._cached_weight = self._get_fused_weight()
        w = self._cached_weight

        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        PAD = self.padding
        H_p = H - 2 * PAD + KH - 1
        W_p = W - 2 * PAD + KW - 1
        H_o = H_p // 2
        W_o = W_p // 2

        pooled = torch.empty((N, OC, H_o, W_o), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_HW = 64
        HW_o = H_o * W_o
        num_hw_tiles = (HW_o + BLOCK_HW - 1) // BLOCK_HW
        num_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (num_hw_tiles, N, num_oc_tiles)
        conv_bn_tanh_pool_kernel[grid](
            x,
            w,
            fused_shift,
            fused_scale,
            pooled,
            N, IC, H, W,
            OC, H_o, W_o,
            PAD=PAD,
            KH=KH,
            KW=KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        G = self.num_groups
        CPG = OC // G

        BLOCK_GN = 1
        while BLOCK_GN < HW_o:
            BLOCK_GN *= 2
        if BLOCK_GN < 16:
            BLOCK_GN = 16

        out = torch.empty_like(pooled)
        grid_gn = (N * G,)
        group_norm_kernel[grid_gn](
            pooled, out,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            N, OC, HW_o,
            G,
            self.group_norm.eps,
            CPG=CPG,
            BLOCK=BLOCK_GN,
            num_warps=4,
            num_stages=2,
        )
        return out