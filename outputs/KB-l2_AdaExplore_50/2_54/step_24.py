import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel_cl(
    x_ptr, mult_ptr, out_ptr,
    total,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = offs % C
    m = tl.load(mult_ptr + c, mask=mask, other=0.0)

    y = x * m
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    tl.store(out_ptr + offs, g, mask=mask)


@triton.jit
def fused_epilogue_kernel(
    x_ptr, mult_ptr, out_ptr,
    total,
    C: tl.constexpr, HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = (offs // HW) % C
    m = tl.load(mult_ptr + c, mask=c < C, other=0.0)

    y = x * m
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    tl.store(out_ptr + offs, g, mask=mask)


def fused_epilogue_cl(x, multiplier):
    # x is channels_last (NHWC-contiguous). Flatten preserving inner C.
    N, C, H, W = x.shape
    out = torch.empty_like(x, memory_format=torch.channels_last)
    total = x.numel()
    BLOCK_SIZE = 2048
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mult_flat = multiplier.contiguous().view(-1)
    fused_epilogue_kernel_cl[grid](
        x, mult_flat, out,
        total,
        C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # convert conv weights to channels_last for faster cuDNN NHWC path
        self.conv = self.conv.to(memory_format=torch.channels_last)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv(x)
        x = fused_epilogue_cl(x, self.multiplier)
        return x