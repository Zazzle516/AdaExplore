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
    # mish(x) = x * tanh(softplus(x))
    # tanh(softplus(x)) = 2*sigmoid(2*softplus(x)) - 1
    # but simpler: use tanh directly via (1 - 2/(exp(2*sp)+1))
    sp1 = tl.log(1.0 + tl.exp(x))
    e1 = tl.exp(2.0 * sp1)
    t1 = 1.0 - 2.0 / (e1 + 1.0)
    y1 = x * t1
    sp2 = tl.log(1.0 + tl.exp(y1))
    e2 = tl.exp(2.0 * sp2)
    t2 = 1.0 - 2.0 / (e2 + 1.0)
    y2 = y1 * t2
    tl.store(out_ptr + offsets, y2, mask=mask)


def double_mish(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    double_mish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv = self.conv.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv(x)
        x = double_mish(x)
        return x