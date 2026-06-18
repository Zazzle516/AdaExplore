import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_conv_bn_tanh_maxpool_gn_kernel(
    x_ptr,           # input: (N, IC, H_in, W_in)
    w_ptr,           # flipped weight: (OC, IC, KH, KW) - regular conv equivalent
    conv_bias_ptr,   # (OC,)
    bn_scale_ptr,    # (OC,)
    bn_bias_ptr,     # (OC,)
    gn_weight_ptr,   # (OC,)
    gn_bias_ptr,     # (OC,)
    out_ptr,         # output: (N, OC, H_pool, W_pool)
    N, IC,
    H_in: tl.constexpr, W_in: tl.constexpr,
    OC: tl.constexpr,
    H_conv: tl.constexpr, W_conv: tl.constexpr,
    H_pool: tl.constexpr, W_pool: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL_POOL: tl.constexpr,  # H_pool * W_pool
    GROUP_SIZE: tl.constexpr,     # CHANNELS_PER_GROUP * SPATIAL_POOL
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    # One program per (n, group). Computes conv->BN->tanh->maxpool->groupnorm
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE

    c_in_group = offs // SPATIAL_POOL
    sp_pool = offs % SPATIAL_POOL
    c = g * CHANNELS_PER_GROUP + c_in_group  # OC channel index

    h_pool = sp_pool // W_pool
    w_pool = sp_pool % W_pool
    # The 2x2 maxpool window in conv-output coords:
    h_conv0 = h_pool * 2
    w_conv0 = w_pool * 2

    # Load BN scale/bias and conv bias per element
    scale = tl.load(bn_scale_ptr + c, mask=mask, other=0.0)
    bn_b = tl.load(bn_bias_ptr + c, mask=mask, other=0.0)
    cb = tl.load(conv_bias_ptr + c, mask=mask, other=0.0)

    # For each of 4 pooled positions, compute conv output. We accumulate
    # in 4 vectors of length BLOCK.
    acc00 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK,), dtype=tl.float32)

    # Regular conv with flipped weight: padding p' = KH-1-PAD.
    PAD_EQ: tl.constexpr = KH - 1 - PAD

    # For each (kh, kw) and each ic, load weight per c (BLOCK long) and the 4 inputs.
    for ic in range(0, IC):
        for kh in range(0, KH):
            for kw in range(0, KW):
                # weight[c, ic, kh, kw]: shape [BLOCK]
                w_off = c * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                wv = tl.load(w_ptr + w_off, mask=mask, other=0.0)

                # Input positions for the 4 pool positions
                # h_in = h_conv + kh - PAD_EQ
                hi00 = h_conv0 + kh - PAD_EQ
                hi10 = (h_conv0 + 1) + kh - PAD_EQ
                wi00 = w_conv0 + kw - PAD_EQ
                wi01 = (w_conv0 + 1) + kw - PAD_EQ

                # Validity
                hi00_v = (hi00 >= 0) & (hi00 < H_in)
                hi10_v = (hi10 >= 0) & (hi10 < H_in)
                wi00_v = (wi00 >= 0) & (wi00 < W_in)
                wi01_v = (wi01 >= 0) & (wi01 < W_in)

                x_base = n * (IC * H_in * W_in) + ic * (H_in * W_in)

                m00 = mask & hi00_v & wi00_v
                m01 = mask & hi00_v & wi01_v
                m10 = mask & hi10_v & wi00_v
                m11 = mask & hi10_v & wi01_v

                x00 = tl.load(x_ptr + x_base + hi00 * W_in + wi00, mask=m00, other=0.0)
                x01 = tl.load(x_ptr + x_base + hi00 * W_in + wi01, mask=m01, other=0.0)
                x10 = tl.load(x_ptr + x_base + hi10 * W_in + wi00, mask=m10, other=0.0)
                x11 = tl.load(x_ptr + x_base + hi10 * W_in + wi01, mask=m11, other=0.0)

                acc00 += wv * x00
                acc01 += wv * x01
                acc10 += wv * x10
                acc11 += wv * x11

    # Apply conv bias + BN affine + tanh
    acc00 = acc00 + cb
    acc01 = acc01 + cb
    acc10 = acc10 + cb
    acc11 = acc11 + cb

    t00 = tl.extra.cuda.libdevice.tanh(acc00 * scale + bn_b)
    t01 = tl.extra.cuda.libdevice.tanh(acc01 * scale + bn_b)
    t10 = tl.extra.cuda.libdevice.tanh(acc10 * scale + bn_b)
    t11 = tl.extra.cuda.libdevice.tanh(acc11 * scale + bn_b)

    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)
    pooled = tl.where(mask, pooled, 0.0)

    sum_val = tl.sum(pooled)
    sumsq_val = tl.sum(pooled * pooled)
    mean = sum_val / GROUP_SIZE
    var = sumsq_val / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    gn_w = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
    gn_b = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)
    result = (pooled - mean) * inv_std * gn_w + gn_b

    out_base = n * (OC * SPATIAL_POOL) + c * SPATIAL_POOL + sp_pool
    tl.store(out_ptr + out_base, result, mask=mask)


@triton.jit
def fused_bn_tanh_maxpool_gn_kernel(
    x_ptr,
    out_ptr,
    bn_scale_ptr,
    bn_bias_ptr,
    gn_weight_ptr,
    gn_bias_ptr,
    N, C, H, W,
    H_out, W_out,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL_OUT: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, BLOCK)
    mask = offs < GROUP_SIZE

    c_in_group = offs // SPATIAL_OUT
    sp_out = offs % SPATIAL_OUT
    c = g * CHANNELS_PER_GROUP + c_in_group

    h_out = sp_out // W_out
    w_out = sp_out % W_out
    h_in_base = h_out * 2
    w_in_base = w_out * 2

    scale = tl.load(bn_scale_ptr + c, mask=mask, other=0.0)
    bias = tl.load(bn_bias_ptr + c, mask=mask, other=0.0)

    base = n * C * H * W + c * H * W
    v00 = tl.load(x_ptr + base + h_in_base * W + w_in_base, mask=mask, other=-1e30)
    v01 = tl.load(x_ptr + base + h_in_base * W + (w_in_base + 1), mask=mask, other=-1e30)
    v10 = tl.load(x_ptr + base + (h_in_base + 1) * W + w_in_base, mask=mask, other=-1e30)
    v11 = tl.load(x_ptr + base + (h_in_base + 1) * W + (w_in_base + 1), mask=mask, other=-1e30)

    t00 = tl.extra.cuda.libdevice.tanh(v00 * scale + bias)
    t01 = tl.extra.cuda.libdevice.tanh(v01 * scale + bias)
    t10 = tl.extra.cuda.libdevice.tanh(v10 * scale + bias)
    t11 = tl.extra.cuda.libdevice.tanh(v11 * scale + bias)

    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)
    pooled = tl.where(mask, pooled, 0.0)

    sum_val = tl.sum(pooled)
    sumsq_val = tl.sum(pooled * pooled)
    mean = sum_val / GROUP_SIZE
    var = sumsq_val / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    gn_w = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
    gn_b = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)

    result = (pooled - mean) * inv_std * gn_w + gn_b

    out_base = n * C * SPATIAL_OUT + c * SPATIAL_OUT + sp_out
    tl.store(out_ptr + out_base, result, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
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
        self.gn_eps = 1e-5

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        PAD = self.padding

        # ConvTranspose2d with stride=1 equivalent dims
        H_conv = H_in + KH - 1 - 2 * PAD
        W_conv = W_in + KW - 1 - 2 * PAD
        H_pool = H_conv // 2
        W_pool = W_conv // 2

        # Fold BN into scale/bias
        bn = self.batch_norm
        bn_scale = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).contiguous()
        bn_bias_folded = (bn.bias - bn.running_mean * bn_scale).contiguous()

        # ConvTranspose weight: (IC, OC, KH, KW). For equivalent regular conv,
        # transpose to (OC, IC, KH, KW) and spatially flip.
        w = self.conv_transpose.weight  # (IC, OC, KH, KW)
        w_eq = w.permute(1, 0, 2, 3).contiguous()  # (OC, IC, KH, KW)
        w_eq = torch.flip(w_eq, dims=[2, 3]).contiguous()

        conv_bias = self.conv_transpose.bias
        if conv_bias is None:
            conv_bias = torch.zeros(OC, device=x.device, dtype=x.dtype)
        conv_bias = conv_bias.contiguous()

        out = torch.empty((N, OC, H_pool, W_pool), device=x.device, dtype=x.dtype)

        groups = self.num_groups
        channels_per_group = OC // groups
        spatial_pool = H_pool * W_pool
        group_size = channels_per_group * spatial_pool

        BLOCK = 1
        while BLOCK < group_size:
            BLOCK *= 2

        if BLOCK <= 256:
            num_warps = 2
        elif BLOCK <= 1024:
            num_warps = 4
        else:
            num_warps = 8

        grid = (N * groups,)

        fused_conv_bn_tanh_maxpool_gn_kernel[grid](
            x, w_eq, conv_bias,
            bn_scale, bn_bias_folded,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            out,
            N, IC,
            H_in=H_in, W_in=W_in,
            OC=OC,
            H_conv=H_conv, W_conv=W_conv,
            H_pool=H_pool, W_pool=W_pool,
            KH=KH, KW=KW, PAD=PAD,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            SPATIAL_POOL=spatial_pool,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.gn_eps,
            num_warps=num_warps,
            num_stages=2,
        )

        return out