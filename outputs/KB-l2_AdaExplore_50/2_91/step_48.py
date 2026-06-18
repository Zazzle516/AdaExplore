import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    SCALE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, hw_tile)
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Pointers: x[n, c, hw] = base + c*HW + hw
    base = pid_n * C * HW
    # 2D ptrs: [BLOCK_C, BLOCK_HW]
    ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    mask2 = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(ptrs, mask=mask2, other=-float('inf'))
    # softmax over channel (axis=0)
    m = tl.max(x, axis=0)  # [BLOCK_HW]
    e = tl.exp(x - m[None, :])
    e = tl.where(mask2, e, 0.0)
    s = tl.sum(e, axis=0)  # [BLOCK_HW]
    soft = e / s[None, :]

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    y = (soft + b[:, None]) * SCALE
    out = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    tl.store(out_ptrs, out, mask=mask2)


def fused_softmax_bias_scale_sigmoid(x, bias, scale):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_HW = 64
    grid = (N, (HW + BLOCK_HW - 1) // BLOCK_HW)
    bias_flat = bias.contiguous().view(-1)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, HW,
        SCALE=float(scale),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)
        return x