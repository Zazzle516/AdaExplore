import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel(
    x_ptr, sum_ptr, out_ptr,
    N, C, D, H, W,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    spatial = D * H * W
    cs = (offsets // spatial) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    s = tl.load(sum_ptr + cs, mask=mask, other=0.0)

    # leaky relu with slope 0.2
    x = tl.where(x >= 0, x, 0.2 * x)
    # add
    x = x + s
    # clamp
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    # gelu (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offsets, x, mask=mask)


def fused_epilogue(x, sum_tensor):
    x = x.contiguous()
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    total = x.numel()
    sum_flat = sum_tensor.contiguous().view(-1)
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, sum_flat, out,
        N, C, D, H, W,
        total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.sum_tensor)
        return x