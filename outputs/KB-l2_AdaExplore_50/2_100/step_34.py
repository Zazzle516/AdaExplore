import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_clamp_div_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, channel_stride, n_channels,
    min_value, inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // channel_stride) % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x = x + b
    x = tl.where(x < min_value, min_value, x)
    x = x * inv_divisor
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_bias_clamp_div(x, bias, min_value, divisor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    N, C, D, H, W = x.shape
    channel_stride = D * H * W
    fused_bias_clamp_div_kernel[grid](
        x, bias, out, n, channel_stride, C,
        float(min_value), float(1.0 / divisor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(conv.bias.detach().clone())
        conv.bias = None
        self.conv_transpose = conv
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = x.contiguous()
        x = self.conv_transpose(x)
        x = fused_bias_clamp_div(x, self.bias, self.min_value, self.divisor)
        return x