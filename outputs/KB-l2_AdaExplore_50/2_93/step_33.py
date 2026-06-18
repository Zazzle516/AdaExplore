import math
import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=4, num_stages=2),
    ],
    key=["n_elements"],
)
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
    x = x + add_value
    # min(x, 0)
    x = tl.minimum(x, 0.0)
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    g = g * multiply_value
    tl.store(out_ptr + offsets, g, mask=mask)


def fused_epilogue(x: torch.Tensor, add_value: float, multiply_value: float) -> torch.Tensor:
    # Treat as flat contiguous memory - layout doesn't matter for elementwise op
    x_flat = x.view(-1) if x.is_contiguous() else x.contiguous().view(-1)
    out = torch.empty_like(x_flat)
    n = x_flat.numel()
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_epilogue_kernel[grid](x_flat, out, n, float(add_value), float(multiply_value))
    return out.view_as(x)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        # Convert weights to channels_last for cuDNN NHWC path
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)
        self.add_value = add_value
        self.multiply_value = multiply_value

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        x = fused_epilogue(x, self.add_value, self.multiply_value)
        return x