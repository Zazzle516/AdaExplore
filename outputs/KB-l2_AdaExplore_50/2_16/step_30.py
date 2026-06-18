import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose2d_fused_kernel(
    x_ptr,        # input  [N, IC, IH, IW]  contiguous NCHW
    w_ptr,        # weight [IC, OC, KH, KW] contiguous
    bias_ptr,     # [OC]
    out_ptr,      # output [N, OC, OH, OW] contiguous NCHW
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    add_value: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oh = sp_offs // OW   # [BLOCK_SP]
    ow = sp_offs % OW    # [BLOCK_SP]

    sp_mask = sp_offs < (OH * OW)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each (kh, kw): compute corresponding input position
    # ih*stride = oh + pad - kh  =>  ih = (oh + pad - kh) / stride if divisible
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = (ih_num >= 0) & (ih_num - ih * STRIDE == 0) & (ih < IH) & (ih >= 0)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = (iw_num >= 0) & (iw_num - iw * STRIDE == 0) & (iw < IW) & (iw >= 0)
            valid = ih_valid & iw_valid  # [BLOCK_SP]

            # Reduction over IC
            ih_safe = tl.where(ih_valid, ih, 0)
            iw_safe = tl.where(iw_valid, iw, 0)

            # input pointer base for this (n, ih, iw) for ic=0
            # offset = ((n*IC + ic)*IH + ih)*IW + iw
            # = n*IC*IH*IW + ic*IH*IW + ih*IW + iw
            base_in = pid_n * IC * IH * IW + ih_safe * IW + iw_safe  # [BLOCK_SP]
            # weight offset: ((ic*OC + oc)*KH + kh)*KW + kw
            base_w = (oc_offs * KH + kh) * KW + kw  # [BLOCK_OC], for ic=0 add ic*OC*KH*KW

            # Loop over IC
            for ic in range(0, IC):
                in_offs = base_in + ic * IH * IW  # [BLOCK_SP]
                w_offs = base_w + ic * OC * KH * KW  # [BLOCK_OC]
                x_vals = tl.load(x_ptr + in_offs, mask=sp_mask & valid, other=0.0)  # [BLOCK_SP]
                w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += x_vals[:, None] * w_vals[None, :]

    # bias
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + b[None, :]

    # Mish: x * tanh(softplus(x))
    ax = tl.abs(acc)
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-ax))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th
    y = y + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale

    # store [BLOCK_SP, BLOCK_OC] into out[n, oc, oh, ow]
    # output offset: ((n*OC + oc)*OH + oh)*OW + ow
    out_offs = (pid_n * OC + oc_offs[None, :]) * (OH * OW) + sp_offs[:, None]
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


def conv_transpose2d_fused(x, weight, bias, stride, padding, output_padding,
                            add_value, scale):
    N, IC, IH, IW = x.shape
    IC2, OC, KH, KW = weight.shape
    assert IC == IC2
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_SP = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))
    conv_transpose2d_fused_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
        float(add_value), float(scale),
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = x.contiguous()
        return conv_transpose2d_fused(
            x,
            self.conv_transpose.weight,
            self.conv_transpose.bias,
            self.stride,
            self.padding,
            self.output_padding,
            self.add_value,
            self.scale,
        )