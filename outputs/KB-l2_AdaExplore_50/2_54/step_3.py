import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel(
    x_ptr, mult_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H * W
    mask = offs < total

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # compute channel index
    hw = H * W
    c = (offs // hw) % C
    m = tl.load(mult_ptr + c, mask=mask, other=0.0)

    y = x * m
    # LeakyReLU with default negative_slope=0.01
    y = tl.where(y >= 0, y, y * 0.01)
    # GELU (exact): 0.5 * y * (1 + erf(y / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    tl.store(out_ptr + offs, g, mask=mask)


def fused_epilogue(x, multiplier):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total = x.numel()
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mult_flat = multiplier.contiguous().view(-1)
    fused_epilogue_kernel[grid](
        x, mult_flat, out,
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.multiplier)
        return x