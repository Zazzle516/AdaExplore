import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr,
    bias_ptr,
    n_elements,
    HW,
    C,
    add_value,
    multiply_value,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index for bias
    c_idx = (offsets // HW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b + add_value
    # min(x, 0)
    x = tl.minimum(x, 0.0)
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    g = g * multiply_value
    tl.store(x_ptr + offsets, g, mask=mask)


def fused_epilogue_inplace(x: torch.Tensor, bias: torch.Tensor, add_value: float, multiply_value: float) -> torch.Tensor:
    if not x.is_contiguous():
        x = x.contiguous()
    n = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_epilogue_kernel[grid](x, bias, n, HW, C, float(add_value), float(multiply_value))
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        # Build the conv with bias, then strip bias from the module so torch
        # doesn't add it; we'll fuse bias into the epilogue kernel instead.
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        with torch.no_grad():
            self.conv_transpose.weight.copy_(conv.weight)
        self.bias = nn.Parameter(conv.bias.detach().clone())
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_epilogue_inplace(x, self.bias, self.add_value, self.multiply_value)
        return x