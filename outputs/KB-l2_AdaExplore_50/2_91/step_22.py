import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor,
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

    m = tl.max(x, axis=0)
    x_shift = x - m
    e = tl.exp(x_shift)
    s = tl.sum(e, axis=0)
    sm = e / s

    b = tl.load(bias_ptr + offs_c, mask=mask, other=0.0)
    y = (sm + b) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    tl.store(out_ptr + base + offs_c * HW, y, mask=mask)


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * HW,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias.view(-1), out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_conv(x, self.bias, self.scaling_factor)
        return x