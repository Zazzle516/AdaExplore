import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [IC, OC, KH, KW]
    bias_ptr,     # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    add_value: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_M: tl.constexpr,  # spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
):
    pid_m = tl.program_id(0)   # spatial tile id
    pid_n = tl.program_id(1)   # OC tile id
    pid_b = tl.program_id(2)   # batch id

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial offsets
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # oc offsets

    oh = offs_m // OW
    ow = offs_m % OW

    mask_m = offs_m < (OH * OW)
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kh, kw and ic
    for kh in tl.static_range(0, KH):
        # ih_num = oh + PAD - kh
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        ih_valid = ((ih_num % STRIDE) == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            iw_valid = ((iw_num % STRIDE) == 0) & (iw >= 0) & (iw < IW)
            spatial_valid = ih_valid & iw_valid & mask_m  # [BLOCK_M]

            # Pointers for x: x[b, ic, ih, iw], iterate ic
            # base offset for this (ih, iw) per spatial position
            ih_safe = tl.where(spatial_valid, ih, 0)
            iw_safe = tl.where(spatial_valid, iw, 0)
            x_spatial_off = ih_safe * IW + iw_safe  # [BLOCK_M]

            # w[ic, oc, kh, kw]: stride oc*KH*KW per ic; KH*KW per oc; +kh*KW+kw
            w_kk_off = kh * KW + kw  # scalar

            # accumulate over ic
            for ic in range(0, IC):
                x_off = pid_b * (IC * IH * IW) + ic * (IH * IW) + x_spatial_off
                x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)  # [BLOCK_M]

                w_off = ic * (OC * KH * KW) + offs_n * (KH * KW) + w_kk_off
                w_val = tl.load(w_ptr + w_off, mask=mask_n, other=0.0)  # [BLOCK_N]

                acc += x_val[:, None] * w_val[None, :]

    # Add bias
    b_val = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc = acc + b_val[None, :]

    # Mish: x * tanh(softplus(x))
    ax = tl.abs(acc)
    sp = tl.maximum(acc, 0.0) + tl.log(1.0 + tl.exp(-ax))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th

    # Add value, hardtanh, scale
    y = y + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale

    # Store: out[b, oc, oh, ow]
    out_off = (pid_b * OC * OH * OW
               + offs_n[None, :] * (OH * OW)
               + offs_m[:, None])
    mask_out = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, y, mask=mask_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = float(add_value)
        self.scale = float(scale)

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        stride = self.stride
        pad = self.padding
        out_pad = self.output_padding
        OH = (IH - 1) * stride - 2 * pad + KH + out_pad
        OW = (IW - 1) * stride - 2 * pad + KW + out_pad

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_N = 64

        grid = (triton.cdiv(OH * OW, BLOCK_M), triton.cdiv(OC, BLOCK_N), N)
        conv_transpose_fused_kernel[grid](
            x, weight, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
            STRIDE=stride,
            PAD=pad,
            add_value=self.add_value,
            scale=self.scale,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return out