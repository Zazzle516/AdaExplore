import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_conv_bn_tanh_maxpool_kernel(
    x_ptr,           # input: (N, IC, H, W)
    w_ptr,           # weight: (IC, OC, KH, KW) for ConvTranspose2d
    out_ptr,         # output: (N, OC, H_out_pool, W_out_pool)
    bn_scale_ptr,    # (OC,)
    bn_bias_ptr,     # (OC,)
    N, IC, H, W,
    OC, H_conv, W_conv,
    H_out, W_out,    # pooled spatial size
    KH: tl.constexpr, KW: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,  # number of pooled output positions per program
):
    # Grid: (N, OC // BLOCK_OC, ceil(H_out * W_out / BLOCK_SP))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]
    sp_mask = sp_offs < (H_out * W_out)

    h_out = sp_offs // W_out  # pooled output h
    w_out = sp_offs % W_out  # pooled output w

    # For each pooled output, we need 4 conv outputs at positions (2*h_out + dh, 2*w_out + dw)
    # ConvTranspose2d with stride=1: out[h, w] = sum over (ic, kh, kw) of
    #     x[ic, h + PAD - kh, w + PAD - kw] * w[ic, oc, kh, kw]
    # where indices must be in [0, H) and [0, W)

    # Load BN scale/bias for the output channels
    bn_s = tl.load(bn_scale_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    bn_b = tl.load(bn_bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    # We'll compute 4 accumulators, one per pool position (dh in 0..1, dw in 0..1)
    # Shape: [BLOCK_OC, BLOCK_SP]
    acc00 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Conv output positions for the 4 pool positions
    # h_conv_d = 2*h_out + dh, w_conv_d = 2*w_out + dw

    for ic in range(0, IC):
        # Load weight slice for this input channel: w[ic, oc_offs, kh, kw]
        # Weight shape: (IC, OC, KH, KW), strides for OC dim is KH*KW
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                # For each pool position, compute input position
                # h_in = (2*h_out + dh) + PAD - kh
                # w_in = (2*w_out + dw) + PAD - kw
                # Then x_val[ic, h_in, w_in] for n=pid_n
                base_in = pid_n * (IC * H * W) + ic * (H * W)

                # dh=0, dw=0
                h_in = 2 * h_out + PAD - kh
                w_in = 2 * w_out + PAD - kw
                m = sp_mask & (h_in >= 0) & (h_in < H) & (w_in >= 0) & (w_in < W)
                x_off = base_in + h_in * W + w_in
                x_val = tl.load(x_ptr + x_off, mask=m, other=0.0)  # [BLOCK_SP]
                acc00 += w_val[:, None] * x_val[None, :]

                # dh=0, dw=1
                w_in1 = w_in + 1
                m1 = sp_mask & (h_in >= 0) & (h_in < H) & (w_in1 >= 0) & (w_in1 < W)
                x_off1 = base_in + h_in * W + w_in1
                x_val1 = tl.load(x_ptr + x_off1, mask=m1, other=0.0)
                acc01 += w_val[:, None] * x_val1[None, :]

                # dh=1, dw=0
                h_in2 = h_in + 1
                m2 = sp_mask & (h_in2 >= 0) & (h_in2 < H) & (w_in >= 0) & (w_in < W)
                x_off2 = base_in + h_in2 * W + w_in
                x_val2 = tl.load(x_ptr + x_off2, mask=m2, other=0.0)
                acc10 += w_val[:, None] * x_val2[None, :]

                # dh=1, dw=1
                m3 = sp_mask & (h_in2 >= 0) & (h_in2 < H) & (w_in1 >= 0) & (w_in1 < W)
                x_off3 = base_in + h_in2 * W + w_in1
                x_val3 = tl.load(x_ptr + x_off3, mask=m3, other=0.0)
                acc11 += w_val[:, None] * x_val3[None, :]

    # Apply BN: y = acc * scale + bias
    s = bn_s[:, None]
    b = bn_b[:, None]
    y00 = acc00 * s + b
    y01 = acc01 * s + b
    y10 = acc10 * s + b
    y11 = acc11 * s + b

    # tanh
    t00 = tl.extra.cuda.libdevice.tanh(y00)
    t01 = tl.extra.cuda.libdevice.tanh(y01)
    t10 = tl.extra.cuda.libdevice.tanh(y10)
    t11 = tl.extra.cuda.libdevice.tanh(y11)

    # max pool
    m0 = tl.maximum(t00, t01)
    m1 = tl.maximum(t10, t11)
    pooled = tl.maximum(m0, m1)  # [BLOCK_OC, BLOCK_SP]

    # Store: out[n, oc, h_out, w_out]
    out_off = pid_n * (OC * H_out * W_out) + oc_offs[:, None] * (H_out * W_out) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, pooled, mask=out_mask)


@triton.jit
def group_norm_kernel(
    x_ptr,           # in/out: (N, C, S)
    out_ptr,
    gn_weight_ptr,   # (C,)
    gn_bias_ptr,     # (C,)
    N, C, S,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
    EPS: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    offs = tl.arange(0, BLOCK)
    base_group = n * C * SPATIAL + g * CHANNELS_PER_GROUP * SPATIAL

    # Pass 1: compute mean / var
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    num_chunks = (GROUP_SIZE + BLOCK - 1) // BLOCK
    for chunk in range(num_chunks):
        idx = chunk * BLOCK + offs
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + base_group + idx, mask=mask, other=0.0)
        sum_val += tl.sum(v)
        sumsq_val += tl.sum(v * v)

    mean = sum_val / GROUP_SIZE
    var = sumsq_val / GROUP_SIZE - mean * mean
    inv_std = 1.0 / tl.sqrt(var + EPS)

    # Pass 2
    for chunk in range(num_chunks):
        idx = chunk * BLOCK + offs
        mask = idx < GROUP_SIZE

        c_in_group = idx // SPATIAL
        c = g * CHANNELS_PER_GROUP + c_in_group

        gn_w = tl.load(gn_weight_ptr + c, mask=mask, other=0.0)
        gn_b = tl.load(gn_bias_ptr + c, mask=mask, other=0.0)

        v = tl.load(x_ptr + base_group + idx, mask=mask, other=0.0)
        normed = (v - mean) * inv_std
        result = normed * gn_w + gn_b
        tl.store(out_ptr + base_group + idx, result, mask=mask)


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
        if self.training or self.stride != 1:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        PAD = self.padding

        # ConvTranspose2d output shape (stride=1):
        # H_conv = H - 1 + KH - 2*PAD = H + KH - 1 - 2*PAD
        H_conv = H + KH - 1 - 2 * PAD
        W_conv = W + KW - 1 - 2 * PAD

        H_out = H_conv // 2
        W_out = W_conv // 2

        # Compute fused BN scale/bias including conv bias
        bn = self.batch_norm
        bn_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        # The conv has its own bias; include it in the BN bias
        conv_bias = self.conv_transpose.bias
        bn_bias = bn.bias - bn.running_mean * bn_scale + conv_bias * bn_scale

        bn_scale = bn_scale.contiguous()
        bn_bias = bn_bias.contiguous()

        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)

        out = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 128

        if OC % BLOCK_OC != 0:
            BLOCK_OC = OC

        SP_TOTAL = H_out * W_out
        # Choose BLOCK_SP based on SP_TOTAL
        if SP_TOTAL >= 256:
            BLOCK_SP = 256
        elif SP_TOTAL >= 128:
            BLOCK_SP = 128
        else:
            BLOCK_SP = 64

        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (SP_TOTAL + BLOCK_SP - 1) // BLOCK_SP)

        fused_conv_bn_tanh_maxpool_kernel[grid](
            x, weight, out,
            bn_scale, bn_bias,
            N, IC, H, W,
            OC, H_conv, W_conv,
            H_out, W_out,
            KH=KH, KW=KW,
            PAD=PAD,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=8,
            num_stages=3,
        )

        # Group norm
        groups = self.num_groups
        channels_per_group = OC // groups
        spatial = H_out * W_out
        group_size = channels_per_group * spatial

        # Choose BLOCK as next pow2 of min(group_size, 2048)
        def _next_pow2(x):
            p = 1
            while p < x:
                p *= 2
            return p
        BLOCK = _next_pow2(min(group_size, 2048))
        if BLOCK < 256:
            BLOCK = 256

        nw = 8 if group_size > 2048 else 4

        out2 = torch.empty_like(out)
        grid2 = (N * groups,)
        group_norm_kernel[grid2](
            out, out2,
            self.group_norm.weight.contiguous(),
            self.group_norm.bias.contiguous(),
            N, OC, spatial,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            SPATIAL=spatial,
            GROUP_SIZE=group_size,
            BLOCK=BLOCK,
            EPS=self.gn_eps,
            num_warps=nw,
            num_stages=2,
        )

        return out2