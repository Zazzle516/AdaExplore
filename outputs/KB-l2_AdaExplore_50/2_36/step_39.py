import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, oc_tile, hw_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oh = offs_hw // OW
    ow = offs_hw % OW
    mask_hw = offs_hw < (OH * OW)
    mask_oc = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # For each output position (oh, ow), accumulate contributions from input positions
    # ConvTranspose: out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, ih, iw] * w[ic, oc, kh, kw]
    # where ih*stride - pad + kh = oh, so ih = (oh + pad - kh) / stride
    for kh in tl.static_range(0, KH):
        ih_num = oh + PAD - kh
        ih = ih_num // STRIDE
        valid_h = (ih_num % STRIDE == 0) & (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw_num = ow + PAD - kw
            iw = iw_num // STRIDE
            valid_w = (iw_num % STRIDE == 0) & (iw >= 0) & (iw < IW)
            valid = valid_h & valid_w & mask_hw  # [BLOCK_HW]

            # Loop over IC
            # weight layout: (IC, OC, KH, KW)
            # input layout: (N, IC, IH, IW)
            for ic in range(0, IC):
                # Load input value [BLOCK_HW]
                x_offset = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                xv = tl.load(x_ptr + x_offset, mask=valid, other=0.0)
                # Load weight [BLOCK_OC]
                w_offset = ic * OC * KH * KW + offs_oc * KH * KW + kh * KW + kw
                wv = tl.load(w_ptr + w_offset, mask=mask_oc, other=0.0)
                acc += wv[:, None] * xv[None, :]

    # add bias
    bv = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc += bv[:, None]

    # store
    out_offset = pid_n * OC * OH * OW + offs_oc[:, None] * OH * OW + offs_hw[None, :]
    mask = mask_oc[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offset, acc, mask=mask)


@triton.jit
def fused_min_sum_gelu_bias_kernel(
    x_ptr, out_ptr, bias_ptr,
    N, C, H, W,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # one program per (n, w)
    pid = tl.program_id(0)
    n = pid // W
    w = pid % W

    offs_c = tl.arange(0, BLOCK_C)
    offs_h = tl.arange(0, BLOCK_H)
    mask_c = offs_c < C
    mask_h = offs_h < H

    # x[n, c, h, w]: offset = n*C*H*W + c*H*W + h*W + w
    base = n * C * H * W + w
    # [BLOCK_C, BLOCK_H]
    ptrs = base + offs_c[:, None] * H * W + offs_h[None, :] * W
    mask = mask_c[:, None] & mask_h[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=float('inf'))
    # min along C
    m = tl.min(x, axis=0)  # [BLOCK_H]
    # zero out invalid h positions
    m = tl.where(mask_h, m, 0.0)
    s = tl.sum(m, axis=0)

    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * s * (1.0 + tl.erf(s * inv_sqrt2))

    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + n * W + w, out)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        # Use PyTorch's conv_transpose2d (highly optimized) for the heavy op
        x = F.conv_transpose2d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            stride=self.stride, padding=self.padding, output_padding=self.output_padding
        )
        N, C, H, W = x.shape
        x = x.contiguous()

        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)
        BLOCK_C = _next_pow2(C)
        BLOCK_H = _next_pow2(H)
        grid = (N * W,)
        fused_min_sum_gelu_bias_kernel[grid](
            x, out, self.bias, N, C, H, W,
            BLOCK_C=BLOCK_C, BLOCK_H=BLOCK_H,
            num_warps=8,
        )
        return out