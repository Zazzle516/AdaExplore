import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel(
    x_ptr, out_ptr,
    n_elements,
    add_value,
    multiply_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # add
    x = x + add_value
    # min(x, 0)
    x = tl.minimum(x, 0.0)
    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    # multiply
    out = gelu * multiply_value
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_epilogue(x: torch.Tensor, add_value: float, multiply_value: float):
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, out, n_elements,
        float(add_value), float(multiply_value),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_epilogue(x, self.add_value, self.multiply_value)
        return x