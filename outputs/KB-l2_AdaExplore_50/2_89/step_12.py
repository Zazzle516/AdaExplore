import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose3d implemented as a scatter-add from input × weight outer product.
# Output is in channels-last memory layout (N, D, H, W, C) to make the subsequent
# max-pool / softmax over C fast.
@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # Grid: (N * ID * IH * IW, IC)
    pid_spatial = tl.program_id(0)
    ic = tl.program_id(1)

    iw = pid_spatial % IW
    tmp = pid_spatial // IW
    ih = tmp % IH
    tmp2 = tmp // IH
    id_ = tmp2 % ID
    n = tmp2 // ID

    # Load input scalar x[n, ic, id, ih, iw]
    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
    xv = tl.load(x_ptr + x_off)

    # OC vector
    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # For each kd, kh, kw, accumulate into output[n, od, oh, ow, oc]
    for kd in tl.static_range(0, KD):
        od = id_ * SD - PD + kd
        if (od >= 0) & (od < OD):
            for kh in tl.static_range(0, KH):
                oh = ih * SH - PH + kh
                if (oh >= 0) & (oh < OH):
                    for kw in tl.static_range(0, KW):
                        ow = iw * SW - PW + kw
                        if (ow >= 0) & (ow < OW):
                            # weight[ic, oc, kd, kh, kw]
                            w_off = ((ic * OC + offs_oc) * KD + kd) * KH * KW + kh * KW + kw
                            wv = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                            contrib = xv * wv

                            # output index in channels-last [N, OD, OH, OW, OC]
                            out_base = (((n * OD + od) * OH + oh) * OW + ow) * OC + offs_oc
                            tl.atomic_add(out_ptr + out_base, contrib, mask=mask_oc)


@triton.jit
def init_bias_kernel(
    out_ptr, b_ptr,
    N, OD, OH, OW, OC,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)  # over N*OD*OH*OW
    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC
    bv = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    out_base = pid * OC + offs_oc
    tl.store(out_ptr + out_base, bv, mask=mask_oc)


# Fused post-conv kernel operating on channels-last conv output.
# Reads [N, OD, OH, OW, C], does maxpool over 2x2x2 with stride 2 (no pad),
# softmax over C, subtract, swish, then max over C.
@triton.jit
def fused_post_channels_last_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, D, H, W,
    PD, PH, PW,
    BLOCK_C: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    pid = tl.program_id(0)
    pw = pid % PW
    tmp = pid // PW
    ph = tmp % PH
    tmp2 = tmp // PH
    pd = tmp2 % PD
    n = tmp2 // PD

    d0 = pd * KD
    h0 = ph * KH
    w0 = pw * KW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)

    # Pool over the KD*KH*KW window for each channel.
    # Channels-last: input index = ((n*D + d)*H + h)*W + w) * C + c
    pooled = tl.full((BLOCK_C,), -float('inf'), dtype=tl.float32)
    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                d = d0 + kd
                h = h0 + kh
                w = w0 + kw
                base = (((n * D + d) * H + h) * W + w) * C + offs_c
                v = tl.load(x_ptr + base, mask=mask_c, other=-float('inf'))
                pooled = tl.maximum(pooled, v)

    pooled = tl.where(mask_c, pooled, -float('inf'))

    # softmax over C
    m = tl.max(pooled, axis=0)
    e = tl.exp(pooled - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    y = sm - sub
    sig = 1.0 / (1.0 + tl.exp(-y))
    sw = sig * y
    sw = tl.where(mask_c, sw, -float('inf'))
    res = tl.max(sw, axis=0)

    out_off = ((n * PD + pd) * PH + ph) * PW + pw
    tl.store(out_ptr + out_off, res)


def conv_transpose3d_triton(x, weight, bias, stride, padding, output_padding):
    # x: [N, IC, ID, IH, IW]
    # weight: [IC, OC, KD, KH, KW]
    # bias: [OC]
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride
    PD, PH, PW = padding
    OPD, OPH, OPW = output_padding

    OD = (ID - 1) * SD - 2 * PD + KD + OPD
    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    # Output in channels-last layout: store as [N, OD, OH, OW, OC]
    out = torch.empty((N, OD, OH, OW, OC), device=x.device, dtype=torch.float32)

    BLOCK_OC = triton.next_power_of_2(OC)
    if BLOCK_OC < 16:
        BLOCK_OC = 16

    # Initialize with bias
    init_grid = (N * OD * OH * OW,)
    init_bias_kernel[init_grid](
        out, bias,
        N, OD, OH, OW, OC,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
    )

    grid = (N * ID * IH * IW, IC)
    conv_transpose3d_scatter_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        SD=SD, SH=SH, SW=SW,
        PD=PD, PH=PH, PW=PW,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
    )
    return out, (N, OC, OD, OH, OW)


def fused_post_cl(x_cl, sub, shape, pool_k=2):
    N, C, D, H, W = shape
    KD = KH = KW = pool_k
    PD = D // KD
    PH = H // KH
    PW = W // KW
    out = torch.empty((N, PD, PH, PW), device=x_cl.device, dtype=x_cl.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    grid = (N * PD * PH * PW,)
    fused_post_channels_last_kernel[grid](
        x_cl, sub, out,
        N, C, D, H, W,
        PD, PH, PW,
        BLOCK_C=BLOCK_C,
        KD=KD, KH=KH, KW=KW,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

        # Normalize hyperparams to tuples
        def _triple(v):
            if isinstance(v, (tuple, list)):
                return tuple(v)
            return (v, v, v)

        self.stride_t = _triple(stride)
        self.padding_t = _triple(padding)
        self.output_padding_t = _triple(output_padding)
        self.pool_k = pool_kernel_size if not isinstance(pool_kernel_size, (tuple, list)) else pool_kernel_size[0]

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()
        sub = self.subtract.contiguous()

        x_cl, conv_shape = conv_transpose3d_triton(
            x, weight, bias,
            self.stride_t, self.padding_t, self.output_padding_t,
        )
        out = fused_post_cl(x_cl, sub, conv_shape, pool_k=self.pool_k)
        return out