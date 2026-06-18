import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_convt_bn_tanh_kernel(
    x_ptr,           # input: (N, IC, H, W)
    w_ptr,           # weight: (IC, OC, KH, KW)
    bias_ptr,        # bias: (OC,) - the actual conv bias
    bn_scale_ptr,    # (OC,)
    bn_bias_ptr,     # (OC,)
    out_ptr,         # output: (N, OC, H_out, W_out)
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program ids: (n, oc_tile, sp_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (H_out * W_out)

    h_out = sp_offs // W_out
    w_out = sp_offs % W_out

    # ConvTranspose: out[n, oc, h, w] = sum_{ic, kh, kw} x[n, ic, h + pad - kh, w + pad - kw] * w[ic, oc, kh, kw]
    # if (h + pad - kh) and (w + pad - kw) in range

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Loop over IC, KH, KW
    for ic in range(0, IC):
        for kh in range(0, KH):
            h_in = h_out + PAD - kh  # shape [BLOCK_SP]
            h_valid = (h_in >= 0) & (h_in < H_in)
            for kw in range(0, KW):
                w_in = w_out + PAD - kw  # shape [BLOCK_SP]
                w_valid = (w_in >= 0) & (w_in < W_in)
                valid = h_valid & w_valid & sp_mask

                # Load input pixel: shape [BLOCK_SP]
                x_offset = pid_n * (IC * H_in * W_in) + ic * (H_in * W_in) + h_in * W_in + w_in
                x_val = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

                # Load weights: w[ic, oc, kh, kw] for all oc in tile -> shape [BLOCK_OC]
                w_offset = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)

                # Outer product
                acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    # Apply BN scale/bias
    bn_s = tl.load(bn_scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    bn_b = tl.load(bn_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc * bn_s[:, None] + bn_b[:, None]

    # Apply tanh
    acc = tl.extra.cuda.libdevice.tanh(acc)

    # Store
    out_offset = pid_n * (OC * H_out * W_out) + oc_offs[:, None] * (H_out * W_out) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offset, acc, mask=out_mask)


@triton.jit
def fused_maxpool_gn_kernel(
    x_ptr,           # input: (N, C, H, W) - after tanh
    out_ptr,         # output: (N, C, H//2, W//2)
    gn_weight_ptr,   # (C,)
    gn_bias_ptr,     # (C,)
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

    base = n * C * H * W + c * H * W
    v00 = tl.load(x_ptr + base + h_in_base * W + w_in_base, mask=mask, other=-1e30)
    v01 = tl.load(x_ptr + base + h_in_base * W + (w_in_base + 1), mask=mask, other=-1e30)
    v10 = tl.load(x_ptr + base + (h_in_base + 1) * W + w_in_base, mask=mask, other=-1e30)
    v11 = tl.load(x_ptr + base + (h_in_base + 1) * W + (w_in_base + 1), mask=mask, other=-1e30)

    m0 = tl.maximum(v00, v01)
    m1 = tl.maximum(v10, v11)
    pooled = tl.maximum(m0, m1)
    pooled = tl.where(mask, pooled, 0.0)

    # Compute mean/var across this group (in registers)
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

        # ConvTranspose2d with stride=1:
        # H_out = H_in - 1 + KH - 2*PAD = H_in + KH - 1 - 2*PAD
        H_out_conv = H_in + KH - 1 - 2 * PAD
        W_out_conv = W_in + KW - 1 - 2 * PAD

        # BN folded scale/bias
        bn = self.batch_norm
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        bn_bias_folded = bn.bias - bn.running_mean * bn_scale
        bn_scale = bn_scale.contiguous()
        bn_bias_folded = bn_bias_folded.contiguous()

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)

        # Intermediate tensor after conv+bn+tanh
        x_conv = torch.empty((N, OC, H_out_conv, W_out_conv), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64
        SP_TOTAL = H_out_conv * W_out_conv

        grid_conv = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (SP_TOTAL + BLOCK_SP - 1) // BLOCK_SP)

        fused_convt_bn_tanh_kernel[grid_conv](
            x, weight, conv_bias,
            bn_scale, bn_bias_folded,
            x_conv,
            N, IC, H_in, W_in,
            OC, H_out_conv, W_out_conv,
            KH=KH, KW=KW, PAD=PAD,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )

        # MaxPool + GroupNorm
        H_pool = H_out_conv // 2
        W_pool = W_out_conv // 2
        out = torch.empty((N, OC, H_pool, W_pool), device=x.device, dtype=x.dtype)

        groups = self.num_groups
        channels_per_group = OC // groups
        spatial_out = H_pool * W_pool
        group_size = channels_per_group * spatial_out

        # Choose BLOCK as next power of 2 >= group_size
        BLOCK = 1
        while BLOCK < group_size:
            BLOCK *= 2

        grid_pool = (N * groups,)
        fused_maxpool_gn_kernel[grid_pool](
            x_conv, out,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            N, OC, H_out_conv, W_out_conv,
            H_pool, W_pool,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            SPATIAL_OUT=spatial_out,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.gn_eps,
            num_warps=4,
        )

        return out