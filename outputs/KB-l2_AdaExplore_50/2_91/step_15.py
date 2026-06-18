import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_gather_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [IC, OC, KH, KW]
    conv_bias_ptr,  # [OC]
    extra_bias_ptr, # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    scaling_factor,
    BLOCK_OW: tl.constexpr,
    OC_C: tl.constexpr,   # power-of-2 >= OC
    IC_C: tl.constexpr,   # IC (constexpr)
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    # one program per (N, OH, OW-block)
    pid = tl.program_id(0)
    ow_blocks = tl.cdiv(OW, BLOCK_OW)
    n = pid // (OH * ow_blocks)
    rem = pid % (OH * ow_blocks)
    oh = rem // ow_blocks
    ow_b = rem % ow_blocks

    offs_ow = ow_b * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    mask_ow = offs_ow < OW

    offs_oc = tl.arange(0, OC_C)  # [OC_C]
    mask_oc = offs_oc < OC

    # accumulator: [OC_C, BLOCK_OW]
    acc = tl.zeros((OC_C, BLOCK_OW), dtype=tl.float32)

    # For ConvTranspose2d:
    # output[oh, ow] = sum_{kh, kw, ic} input[ic, ih, iw] * weight[ic, oc, kh, kw]
    # where ih * stride - pad + kh = oh => ih = (oh + pad - kh) / stride
    #       iw * stride - pad + kw = ow => iw = (ow + pad - kw) / stride
    # valid iff (oh + pad - kh) % stride == 0 and 0 <= ih < IH (similarly w)

    # Precompute h-side once
    for kh in tl.static_range(0, KH_C):
        h_num = oh + PAD_H - kh
        ih = h_num // STRIDE_H
        h_valid = (h_num >= 0) & ((h_num % STRIDE_H) == 0) & (ih >= 0) & (ih < IH)

        for kw in tl.static_range(0, KW_C):
            w_num = offs_ow + PAD_W - kw  # [BLOCK_OW]
            iw = w_num // STRIDE_W
            w_valid = (w_num >= 0) & ((w_num % STRIDE_W) == 0) & (iw >= 0) & (iw < IW)
            valid = h_valid & w_valid & mask_ow  # [BLOCK_OW]

            # Load input[n, :, ih, iw] -> [IC_C, BLOCK_OW]
            # x_ptr offset: n*IC*IH*IW + ic*IH*IW + ih*IW + iw
            ic_offs = tl.arange(0, IC_C)
            x_base = n * IC * IH * IW + ih * IW
            x_ptrs = x_ptr + x_base + ic_offs[:, None] * (IH * IW) + iw[None, :]
            x_mask = valid[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [IC_C, BLOCK_OW]

            # Load weight[:, :, kh, kw] -> [IC_C, OC_C]
            # w layout: [IC, OC, KH, KW], offset = ic*OC*KH*KW + oc*KH*KW + kh*KW + kw
            w_ptrs = w_ptr + ic_offs[:, None] * (OC * KH * KW) + offs_oc[None, :] * (KH * KW) + kh * KW + kw
            w_mask = mask_oc[None, :]
            w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [IC_C, OC_C]

            # acc[OC_C, BLOCK_OW] += w_vals.T @ x_vals
            acc += tl.dot(tl.trans(w_vals), x_vals)

    # Add conv bias
    cb = tl.load(conv_bias_ptr + offs_oc, mask=mask_oc, other=0.0)  # [OC_C]
    acc += cb[:, None]

    # Mask invalid OC rows to -inf for softmax
    acc = tl.where(mask_oc[:, None] & mask_ow[None, :], acc, -float('inf'))

    # Softmax along OC (axis=0)
    max_val = tl.max(acc, axis=0)  # [BLOCK_OW]
    x_shift = acc - max_val[None, :]
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(mask_oc[:, None] & mask_ow[None, :], exp_x, 0.0)
    sum_val = tl.sum(exp_x, axis=0)  # [BLOCK_OW]
    sm = exp_x / sum_val[None, :]

    # Add extra bias [OC, 1, 1]
    eb = tl.load(extra_bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    y = (sm + eb[:, None]) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    # Store: out[n, :, oh, offs_ow]
    out_base = n * OC * OH * OW + oh * OW
    out_ptrs = out_ptr + out_base + offs_oc[:, None] * (OH * OW) + offs_ow[None, :]
    out_mask = mask_oc[:, None] & mask_ow[None, :]
    tl.store(out_ptrs, y, mask=out_mask)


def fused_conv_transpose_softmax_bias_scale_sigmoid(
    x, weight, conv_bias, extra_bias, scaling_factor,
    stride, padding, output_padding,
):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w

    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    x = x.contiguous()
    weight = weight.contiguous()
    extra_bias_flat = extra_bias.contiguous().view(-1)
    conv_bias_flat = conv_bias.contiguous().view(-1)

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    OC_C = 1
    while OC_C < OC:
        OC_C *= 2

    BLOCK_OW = 32
    ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    grid = (N * OH * ow_blocks,)

    conv_transpose_gather_kernel[grid](
        x, weight, conv_bias_flat, extra_bias_flat, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, stride,
        padding, padding,
        float(scaling_factor),
        BLOCK_OW=BLOCK_OW,
        OC_C=OC_C,
        IC_C=IC,
        KH_C=KH,
        KW_C=KW,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        return fused_conv_transpose_softmax_bias_scale_sigmoid(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.bias, self.scaling_factor,
            self.stride, self.padding, self.output_padding,
        )