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
    N, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    # Each program handles BLOCK_HW spatial positions for one batch
    # pid_nhw indexes (n, hw_block)
    blocks_per_n = tl.cdiv(HW, BLOCK_HW)
    n = pid_nhw // blocks_per_n
    hw_block = pid_nhw % blocks_per_n
    hw_start = hw_block * BLOCK_HW

    offs_hw = hw_start + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    mask_hw = offs_hw < HW

    offs_c = tl.arange(0, BLOCK_C)  # [BLOCK_C]
    mask_c = offs_c < C

    # x layout: [N, C, H, W] -> index = n*C*HW + c*HW + hw
    base = n * C * HW
    # ptrs: [BLOCK_C, BLOCK_HW]
    x_ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    mask_2d = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptrs, mask=mask_2d, other=-float('inf'))

    # softmax along channel (axis=0)
    max_val = tl.max(x, axis=0)  # [BLOCK_HW]
    x_shift = x - max_val[None, :]
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(mask_2d, exp_x, 0.0)
    sum_val = tl.sum(exp_x, axis=0)  # [BLOCK_HW]
    sm = exp_x / sum_val[None, :]

    # bias is [C, 1, 1]
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)  # [BLOCK_C]

    y = (sm + bias[:, None]) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, y, mask=mask_2d)


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    HW = H * W

    # Pick BLOCK_C as next power of 2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    BLOCK_HW = 64
    blocks_per_n = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid = (N * blocks_per_n,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
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