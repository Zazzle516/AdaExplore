import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PADDING: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ih, iw); loops over IC and accumulates into OC tile
    pid = tl.program_id(0)
    pid_oc = tl.program_id(1)

    iw = pid % IW
    tmp = pid // IW
    ih = tmp % IH
    n = tmp // IH

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # For each kernel position, accumulate input * weight, then scatter add to output
    # output position: oh = ih*stride - padding + kh, ow = iw*stride - padding + kw
    for kh in tl.static_range(0, KH):
        oh = ih * STRIDE - PADDING + kh
        valid_h = (oh >= 0) & (oh < OH)
        for kw in tl.static_range(0, KW):
            ow = iw * STRIDE - PADDING + kw
            valid_w = (ow >= 0) & (ow < OW)
            valid = valid_h & valid_w

            # accumulate over IC
            acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
            for ic in range(0, IC):
                x_val = tl.load(x_ptr + n * IC * IH * IW + ic * IH * IW + ih * IW + iw)
                # weight layout: (IC, OC, KH, KW)
                w_ptrs = w_ptr + ic * OC * KH * KW + offs_oc * KH * KW + kh * KW + kw
                w_vals = tl.load(w_ptrs, mask=mask_oc, other=0.0)
                acc += x_val * w_vals

            out_ptrs = out_ptr + n * OC * OH * OW + offs_oc * OH * OW + oh * OW + ow
            tl.atomic_add(out_ptrs, acc, mask=mask_oc & valid)


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    base = n * C * HW + hw
    ptrs = x_ptr + base + offs_c * HW

    x = tl.load(ptrs, mask=mask, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    soft = e / s

    b = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)
    y = (soft + b) * SCALE
    out = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c * HW
    tl.store(out_ptrs, out, mask=mask)


def fused_softmax_bias_scale_sigmoid(x, bias, scale):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * HW,)
    bias_flat = bias.contiguous().view(-1)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, HW,
        SCALE=float(scale),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)
        return x