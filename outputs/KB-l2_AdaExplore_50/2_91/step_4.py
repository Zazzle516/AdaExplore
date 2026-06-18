import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    HW = H * W
    n = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W

    base = n * C * HW + h * W + w

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    x_ptrs = x_ptr + base + offs_c * HW
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

    out_ptrs = out_ptr + base + offs_c * HW
    tl.store(out_ptrs, y, mask=mask)


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * H * W,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, H, W,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
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

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post_conv(x, self.bias, self.scaling_factor)
        return x