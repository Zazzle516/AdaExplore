import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr,
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
    c_idx = (offsets // HW) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + bias + add_value
    x = tl.minimum(x, 0.0)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    out = gelu * multiply_value
    tl.store(x_ptr + offsets, out, mask=mask)


def fused_epilogue_inplace(x: torch.Tensor, bias: torch.Tensor, add_value: float, multiply_value: float) -> torch.Tensor:
    x = x.contiguous()
    n = x.numel()
    N, C, H, W = x.shape
    HW = H * W
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_epilogue_kernel[grid](
        x, bias, n, HW, C,
        float(add_value), float(multiply_value),
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        bias = self.conv_transpose.bias
        x = torch.nn.functional.conv_transpose2d(
            x, self.conv_transpose.weight, bias=None,
            stride=self.conv_transpose.stride,
            padding=self.conv_transpose.padding,
            output_padding=self.conv_transpose.output_padding,
            dilation=self.conv_transpose.dilation,
            groups=self.conv_transpose.groups,
        )
        x = fused_epilogue_inplace(x, bias, self.add_value, self.multiply_value)
        return x