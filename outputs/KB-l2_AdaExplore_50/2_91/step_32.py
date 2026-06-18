import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 4}, num_warps=2, num_stages=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    num_hw_blocks = (HW + BLOCK_HW - 1) // BLOCK_HW
    n = pid // num_hw_blocks
    hw_blk = pid % num_hw_blocks
    hw_start = hw_blk * BLOCK_HW

    offs_c = tl.arange(0, BLOCK_C)
    offs_hw = hw_start + tl.arange(0, BLOCK_HW)

    mask_c = offs_c < C
    mask_hw = offs_hw < HW
    mask = mask_c[:, None] & mask_hw[None, :]

    base = n * C * HW
    x_ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]

    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    m = tl.max(x, axis=0)
    x_shift = x - m[None, :]
    e = tl.exp(x_shift)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    y = (sm + b[:, None]) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    tl.store(out_ptr + base + offs_c[:, None] * HW + offs_hw[None, :], y, mask=mask)


def fused_post_conv(x, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK_C = triton.next_power_of_2(C)
    grid = lambda meta: (N * triton.cdiv(HW, meta['BLOCK_HW']),)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias.view(-1), out,
        N, C, HW,
        scaling_factor=float(scaling_factor),
        BLOCK_C=BLOCK_C,
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