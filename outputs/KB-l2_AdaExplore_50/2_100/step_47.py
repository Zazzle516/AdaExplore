import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_clamp_div_kernel(
    x_ptr, bias_ptr, out_ptr,
    spatial_size, n_channels,
    BLOCK_SIZE: tl.constexpr,
    MIN_VALUE: tl.constexpr,
    INV_DIVISOR: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    c = pid_nc % n_channels
    b = tl.load(bias_ptr + c)
    base = pid_nc * spatial_size
    offs = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < spatial_size
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    x = x + b
    x = tl.where(x < MIN_VALUE, MIN_VALUE, x)
    x = x * INV_DIVISOR
    tl.store(out_ptr + base + offs, x, mask=mask)


def fused_bias_clamp_div(x, bias, min_value, divisor):
    x = x.contiguous()
    out = torch.empty_like(x)
    N, C, D, H, W = x.shape
    spatial = D * H * W
    BLOCK_SIZE = 1024
    grid = (N * C, (spatial + BLOCK_SIZE - 1) // BLOCK_SIZE)
    fused_bias_clamp_div_kernel[grid](
        x, bias, out, spatial, C,
        BLOCK_SIZE=BLOCK_SIZE,
        MIN_VALUE=float(min_value),
        INV_DIVISOR=float(1.0 / divisor),
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        torch.backends.cudnn.benchmark = True
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