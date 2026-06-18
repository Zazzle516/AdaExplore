import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel_nhwc(
    x_ptr,
    bias_ptr,
    out_ptr,
    NHW, C,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * C

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    x_ptrs = x_ptr + base + offs_c
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    max_val = tl.max(x, axis=0)
    x_shift = x - max_val
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(mask, exp_x, 0.0)
    sum_val = tl.sum(exp_x, axis=0)
    inv_sum = 1.0 / sum_val
    sm = exp_x * inv_sum

    bias = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)

    y = (sm + bias) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c
    tl.store(out_ptrs, y, mask=mask)


def fused_post_conv(x, bias, scaling_factor):
    # x is in channels_last memory format: NCHW logical, NHWC physical
    N, C, H, W = x.shape
    # Reinterpret as contiguous NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    bias_flat = bias.contiguous().view(-1)
    out_nhwc = torch.empty_like(x_nhwc)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    NHW = N * H * W
    grid = (NHW,)
    fused_softmax_bias_scale_sigmoid_kernel_nhwc[grid](
        x_nhwc, bias_flat, out_nhwc,
        NHW, C,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=2,
        num_stages=2,
    )
    return out_nhwc.permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_conv(x, self.bias, self.scaling_factor)
        return x