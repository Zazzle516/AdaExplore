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
    # mish: x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
    sp1 = tl.log(1.0 + tl.exp(x))
    y1 = x * ((tl.exp(sp1) - tl.exp(-sp1)) / (tl.exp(sp1) + tl.exp(-sp1)))
    sp2 = tl.log(1.0 + tl.exp(y1))
    y2 = y1 * ((tl.exp(sp2) - tl.exp(-sp2)) / (tl.exp(sp2) + tl.exp(-sp2)))
    tl.store(out_ptr + offsets, y2, mask=mask)


def double_mish(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    double_mish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = double_mish(x)
        return x