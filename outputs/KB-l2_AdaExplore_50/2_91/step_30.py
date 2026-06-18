import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C: tl.constexpr, HW,
    scaling_factor: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, num_hw_blocks)
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    offs_c = tl.arange(0, C)

    base = pid_n * C * HW
    # x layout: [N, C, HW]; we want tile [C, BLOCK_HW]
    x_ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    x = tl.load(x_ptrs, mask=mask_hw[None, :], other=-float('inf'))

    # softmax over C
    m = tl.max(x, axis=0)  # [BLOCK_HW]
    x_shift = x - m[None, :]
    e = tl.exp(x_shift)
    s = tl.sum(e, axis=0)  # [BLOCK_HW]
    sm = e / s[None, :]

    b = tl.load(bias_ptr + offs_c, mask=offs_c < C, other=0.0)  # [C]
    y = (sm + b[:, None]) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, y, mask=mask_hw[None, :])


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK_HW = 64
    grid = (N, triton.cdiv(HW, BLOCK_HW))
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias.view(-1), out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
        num_stages=3,
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
        y = F.conv_transpose2d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding,
        )
        y = y.contiguous()
        return fused_post_conv(y, self.bias, self.scaling_factor)