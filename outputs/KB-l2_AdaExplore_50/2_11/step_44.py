import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_bn_tanh_pool_kernel(
    x_ptr,       # input [N, IC, H, W]
    w_ptr,       # conv_transpose weight [IC, OC, KH, KW] (PyTorch storage)
    bias_ptr,    # fused bias [OC]
    scale_ptr,   # fused scale [OC]
    out_ptr,     # output after pool [N, OC, H_out, W_out]
    N, IC, H, W,
    OC, H_p, W_p,
    H_o, W_o,    # pooled output dims
    PAD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,  # number of pooled output positions per program
):
    pid = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_oc = tl.program_id(2)

    # pid covers tiles over pooled output positions
    hw_start = pid * BLOCK_HW
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)  # pooled positions
    mask_hw = offs_hw < (H_o * W_o)

    oh_p = (offs_hw // W_o) * 2  # pre-pool h coordinate (top-left of 2x2)
    ow_p = (offs_hw % W_o) * 2

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    mask_oc = oc_offs < OC

    # Accumulators for 4 corners of 2x2 pool window: shape [BLOCK_OC, BLOCK_HW]
    acc00 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # ConvTranspose2d with stride=1: equivalent to conv with flipped kernel and padding KH-1-PAD
    # output[n, oc, oh, ow] = sum_ic sum_kh sum_kw input[n, ic, oh+kh-(KH-1-pad), ow+kw-(KW-1-pad)] * weight[ic, oc, KH-1-kh, KW-1-kw]
    # Using flipped kernel indexing: let kh' = KH-1-kh, kw' = KW-1-kw
    # output[n,oc,oh,ow] = sum_ic sum_kh' sum_kw' input[n, ic, oh+(KH-1-kh')-(KH-1-pad), ow+(KW-1-kw')-(KW-1-pad)] * weight[ic, oc, kh', kw']
    # = sum_ic sum_kh' sum_kw' input[n, ic, oh - kh' + pad, ow - kw' + pad] * weight[ic, oc, kh', kw']
    # So effective input index: ih = oh + pad - kh', iw = ow + pad - kw'
    PAD_EFF = PAD  # padding from original conv_transpose

    # Loop over 4 pool corners (dh, dw) in {0,1}
    # We process all 4 in parallel by tracking 4 accumulators
    # ih for corner (dh, dw) at kernel pos (kh, kw): (oh_p + dh) + PAD - kh
    # We iterate over kh, kw
    n_off = pid_n * IC * H * W

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # weight pointer for [ic, oc, kh, kw] across all ic, oc tile
            # weight shape: [IC, OC, KH, KW], stride_ic = OC*KH*KW, stride_oc = KH*KW
            # We need w[ic, oc_offs, kh, kw] -> shape [IC, BLOCK_OC]
            
            ih00 = oh_p + PAD_EFF - kh
            iw00 = ow_p + PAD_EFF - kw
            ih01 = oh_p + PAD_EFF - kh
            iw01 = ow_p + 1 + PAD_EFF - kw
            ih10 = oh_p + 1 + PAD_EFF - kh
            iw10 = ow_p + PAD_EFF - kw
            ih11 = oh_p + 1 + PAD_EFF - kh
            iw11 = ow_p + 1 + PAD_EFF - kw

            valid00 = (ih00 >= 0) & (ih00 < H) & (iw00 >= 0) & (iw00 < W)
            valid01 = (ih01 >= 0) & (ih01 < H) & (iw01 >= 0) & (iw01 < W)
            valid10 = (ih10 >= 0) & (ih10 < H) & (iw10 >= 0) & (iw10 < W)
            valid11 = (ih11 >= 0) & (ih11 < H) & (iw11 >= 0) & (iw11 < W)

            # Accumulate over IC
            for ic in range(0, IC):
                # Load weight[ic, oc_offs, kh, kw] -> [BLOCK_OC]
                w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                ic_off = n_off + ic * H * W

                # Corner 00
                x_off00 = ic_off + ih00 * W + iw00
                x00 = tl.load(x_ptr + x_off00, mask=valid00 & mask_hw, other=0.0)  # [BLOCK_HW]
                acc00 += w_val[:, None] * x00[None, :]

                x_off01 = ic_off + ih01 * W + iw01
                x01 = tl.load(x_ptr + x_off01, mask=valid01 & mask_hw, other=0.0)
                acc01 += w_val[:, None] * x01[None, :]

                x_off10 = ic_off + ih10 * W + iw10
                x10 = tl.load(x_ptr + x_off10, mask=valid10 & mask_hw, other=0.0)
                acc10 += w_val[:, None] * x10[None, :]

                x_off11 = ic_off + ih11 * W + iw11
                x11 = tl.load(x_ptr + x_off11, mask=valid11 & mask_hw, other=0.0)
                acc11 += w_val[:, None] * x11[None, :]

    # Apply fused BN: y = scale * (acc + conv_bias) + bn_shift, but we fold conv_bias into bias_ptr fully
    # bias_ptr is the final shift; scale_ptr is the scale
    scale = tl.load(scale_ptr + oc_offs, mask=mask_oc, other=0.0)  # [BLOCK_OC]
    shift = tl.load(bias_ptr + oc_offs, mask=mask_oc, other=0.0)   # [BLOCK_OC]

    v00 = acc00 * scale[:, None] + shift[:, None]
    v01 = acc01 * scale[:, None] + shift[:, None]
    v10 = acc10 * scale[:, None] + shift[:, None]
    v11 = acc11 * scale[:, None] + shift[:, None]

    # tanh
    t00 = 2.0 * tl.sigmoid(2.0 * v00) - 1.0
    t01 = 2.0 * tl.sigmoid(2.0 * v01) - 1.0
    t10 = 2.0 * tl.sigmoid(2.0 * v10) - 1.0
    t11 = 2.0 * tl.sigmoid(2.0 * v11) - 1.0

    # max pool 2x2
    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)  # [BLOCK_OC, BLOCK_HW]

    # Store to output [N, OC, H_o, W_o]
    out_off = pid_n * OC * H_o * W_o + oc_offs[:, None] * (H_o * W_o) + offs_hw[None, :]
    store_mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_off, pooled, mask=store_mask)


@triton.jit
def group_norm_kernel(
    x_ptr,        # [N, C, H, W]
    out_ptr,      # [N, C, H, W]
    weight_ptr,   # [C]
    bias_ptr,     # [C]
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

    def forward(self, x):
        if self.training or self.stride != 1:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        bn = self.batch_norm
        # Fold BN into per-channel scale and shift applied to conv output
        # bn_scale = bn.weight / sqrt(bn.running_var + eps)
        # bn_shift = bn.bias - bn.running_mean * bn_scale
        # The fused output is: bn_scale * (conv_out) + bn_shift
        # conv_out already includes conv_transpose.bias. We absorb conv bias into shift:
        # bn_scale * (acc + conv_bias) + bn_shift = bn_scale * acc + (bn_scale * conv_bias + bn_shift)
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bn_shift = bn.bias - bn.running_mean * bn_scale
        fused_shift = bn_scale * self.conv_transpose.bias + bn_shift
        fused_scale = bn_scale

        x = x.contiguous()
        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        PAD = self.padding
        # ConvTranspose2d output size (stride=1): H_p = H + 2*(KH-1) - 2*PAD... wait
        # Actually: H_p = (H-1)*stride - 2*pad + (KH-1) + 1 = H - 2*pad + KH - 1
        H_p = H - 2 * PAD + KH - 1
        W_p = W - 2 * PAD + KW - 1
        H_o = H_p // 2
        W_o = W_p // 2

        pooled = torch.empty((N, OC, H_o, W_o), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_HW = 64
        # H_o * W_o = 16*16 = 256
        HW_o = H_o * W_o
        num_hw_tiles = (HW_o + BLOCK_HW - 1) // BLOCK_HW
        num_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (num_hw_tiles, N, num_oc_tiles)
        conv_bn_tanh_pool_kernel[grid](
            x,
            self.conv_transpose.weight,
            fused_shift.contiguous(),
            fused_scale.contiguous(),
            pooled,
            N, IC, H, W,
            OC, H_p, W_p,
            H_o, W_o,
            PAD=PAD,
            KH=KH,
            KW=KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        # GroupNorm
        G = self.num_groups
        CPG = OC // G
        HW_o = H_o * W_o

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