import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr,
    bias_scaled_ptr,
    out_ptr,
    N, C, HW,
    scaling_factor: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_nhw = tl.program_id(0)
    blocks_per_n = tl.cdiv(HW, BLOCK_HW)
    n = pid_nhw // blocks_per_n
    hw_block = pid_nhw % blocks_per_n
    hw_start = hw_block * BLOCK_HW

    offs_hw = hw_start + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * HW
    x_ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    mask_2d = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptrs, mask=mask_2d, other=-float('inf'))

    max_val = tl.max(x, axis=0)
    x_shift = x - max_val[None, :]
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(mask_2d, exp_x, 0.0)
    sum_val = tl.sum(exp_x, axis=0)
    inv_sum = 1.0 / sum_val
    sm = exp_x * inv_sum[None, :]

    bias_s = tl.load(bias_scaled_ptr + offs_c, mask=mask_c, other=0.0)

    y = sm * scaling_factor + bias_s[:, None]
    y = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, y, mask=mask_2d)


def fused_post_conv(x, bias_scaled, scaling_factor):
    N, C, H, W = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    bias_flat = bias_scaled.contiguous().view(-1)
    out = torch.empty_like(x)
    HW = H * W

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    BLOCK_HW = 512
    blocks_per_n = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid = (N * blocks_per_n,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, HW,
        scaling_factor=float(scaling_factor),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
        num_stages=3,
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
        bias_scaled = self.bias * self.scaling_factor
        x = fused_post_conv(x, bias_scaled, self.scaling_factor)
        return x