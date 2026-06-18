import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def double_mish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # First mish: x * tanh(softplus(x)), numerically stable softplus
    sp1 = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    t1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    y = x * t1
    sp2 = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(y)))
    t2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    z = y * t2
    tl.store(out_ptr + offsets, z, mask=mask)


def double_mish(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 4096
    grid = ((n + BLOCK - 1) // BLOCK,)
    double_mish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Convert to channels_last for faster conv on modern GPUs
        self.conv = self.conv.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv(x)
        x = double_mish(x)
        return x