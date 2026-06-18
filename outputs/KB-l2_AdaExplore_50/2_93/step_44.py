import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
    ],
    key=['n_elements', 'HW', 'C'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements,
    HW, C,
    add_value: tl.constexpr,
    multiply_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # determine channel index for bias: (offset / HW) % C
    c_idx = (offsets // HW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b + add_value
    x = tl.minimum(x, 0.0)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    out = gelu * multiply_value
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor, add_value: float, multiply_value: float) -> torch.Tensor:
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    n = x.numel()
    grid = lambda META: ((n + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'],)
    fused_epilogue_kernel[grid](
        x, bias, x, n,
        HW, C,
        add_value=float(add_value),
        multiply_value=float(multiply_value),
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        # Run conv without bias add (we fuse bias into the epilogue)
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        x = torch.nn.functional.conv_transpose2d(
            x, weight, bias=None,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            groups=self.conv_transpose.groups,
            dilation=self.conv_transpose.dilation,
        )
        x = fused_epilogue(x, bias, self.add_value, self.multiply_value)
        return x