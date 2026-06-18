import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N * num_hw_blocks, num_oc_blocks, IC)
    pid_nhw = tl.program_id(0)
    pid_oc = tl.program_id(1)
    ic = tl.program_id(2)

    num_hw_blocks = tl.cdiv(IH * IW, BLOCK_HW)
    n = pid_nhw // num_hw_blocks
    hw_block = pid_nhw % num_hw_blocks

    offs_hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < (IH * IW)
    ih = offs_hw // IW
    iw = offs_hw % IW

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # load input[n, ic, ih, iw]: shape [BLOCK_HW]
    x_off = ((n * IC + ic) * IH + ih) * IW + iw
    x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)  # [BLOCK_HW]

    # weight is [IC, OC, KH, KW]
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            w_off = ((ic * OC + offs_oc) * KH + kh) * KW + kw  # [BLOCK_OC]
            w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

            # outer product: [BLOCK_HW, BLOCK_OC]
            contrib = x_val[:, None] * w_val[None, :]

            oh = ih * STRIDE - PAD + kh
            ow = iw * STRIDE - PAD + kw

            valid_h = (oh >= 0) & (oh < OH)
            valid_w = (ow >= 0) & (ow < OW)
            valid_hw = valid_h & valid_w & mask_hw  # [BLOCK_HW]

            out_off = ((n * OC + offs_oc[None, :]) * OH + oh[:, None]) * OW + ow[:, None]
            mask = valid_hw[:, None] & mask_oc[None, :]
            tl.atomic_add(out_ptr + out_off, contrib, mask=mask)


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, conv_bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    base = n * C * HW + hw
    x_ptrs = x_ptr + base + offs_c * HW

    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    cb = tl.load(conv_bias_ptr + offs_c, mask=mask, other=0.0)
    x = x + cb

    m = tl.max(x, axis=0)
    x_shift = x - m
    e = tl.exp(x_shift)
    s = tl.sum(e, axis=0)
    sm = e / s

    b = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)
    y = (sm + b) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    tl.store(out_ptr + base + offs_c * HW, y, mask=mask)


def conv_transpose2d_triton(x, weight, conv_bias, stride, padding, output_padding):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    assert IC == IC_w
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.zeros((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 64
    BLOCK_HW = 64
    num_hw_blocks = (IH * IW + BLOCK_HW - 1) // BLOCK_HW
    num_oc_blocks = (OC + BLOCK_OC - 1) // BLOCK_OC

    grid = (N * num_hw_blocks, num_oc_blocks, IC)
    conv_transpose2d_scatter_kernel[grid](
        x, weight, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride, padding,
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
        num_stages=2,
    )
    return out, OH, OW


def fused_post_conv(x, conv_bias, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * HW,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias.view(-1), conv_bias.view(-1), out,
        N, C, HW,
        scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias
        y, OH, OW = conv_transpose2d_triton(
            x, weight, conv_bias,
            self.stride, self.padding, self.output_padding
        )
        out = fused_post_conv(y, conv_bias, self.bias, self.scaling_factor)
        return out