import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'C'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, mult_ptr, out_ptr,
    HW, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid_block = tl.program_id(0)
    pid_nc = tl.program_id(1)

    c = pid_nc % C
    m = tl.load(mult_ptr + c)

    base = pid_nc * HW
    offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    y = x * m
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    tl.store(out_ptr + base + offs, gelu, mask=mask)


def fused_epilogue(x, multiplier):
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: ((HW + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"], N * C)
    fused_epilogue_kernel[grid](
        x, multiplier.contiguous().view(-1), out,
        HW, C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.multiplier)
        return x