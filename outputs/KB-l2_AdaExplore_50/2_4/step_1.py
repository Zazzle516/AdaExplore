import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _double_mish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # First mish: x * tanh(softplus(x))
    sp1 = tl.log(1.0 + tl.exp(x))
    # tanh via sigmoid: tanh(a) = 2*sigmoid(2a) - 1
    t1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    y = x * t1
    sp2 = tl.log(1.0 + tl.exp(y))
    t2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    z = y * t2
    tl.store(out_ptr + offsets, z, mask=mask)


def double_mish(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    _double_mish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        x = double_mish(x)
        return x