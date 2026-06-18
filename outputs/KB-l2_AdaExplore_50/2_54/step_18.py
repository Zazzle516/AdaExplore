import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

# Enable fast conv paths
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['total'],
)
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

    x = tl.load(x_ptr + offs, mask=mask, other=0.0, eviction_policy='evict_first')
    c = (offs // HW) % C
    m = tl.load(mult_ptr + c, mask=mask, other=0.0)

    y = x * m
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    tl.store(out_ptr + offs, g, mask=mask)


def fused_epilogue(x, multiplier):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total = x.numel()
    HW = H * W
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    mult_flat = multiplier.contiguous().view(-1)
    fused_epilogue_kernel[grid](
        x, mult_flat, out,
        total,
        C, HW,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv(x)
        x = fused_epilogue(x, self.multiplier)
        return x