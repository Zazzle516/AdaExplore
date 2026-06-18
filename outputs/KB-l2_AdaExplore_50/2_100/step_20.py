import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_clamp_div_kernel_cl(
    x_ptr, bias_ptr, out_ptr, n_elements, C,
    min_value, inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channels_last_3d => channel is innermost contiguous dim
    c_idx = offsets % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b
    x = tl.where(x < min_value, min_value, x)
    x = x * inv_divisor
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.jit
def fused_clamp_div_kernel(
    x_ptr, out_ptr, n_elements,
    min_value, inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.where(x < min_value, min_value, x)
    x = x * inv_divisor
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_bias_clamp_div_cl(x, bias, min_value, divisor):
    # x is in channels_last_3d format, shape (N, C, D, H, W)
    out = torch.empty_like(x, memory_format=torch.channels_last_3d)
    n = x.numel()
    C = x.shape[1]
    BLOCK_SIZE = 2048
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bias_clamp_div_kernel_cl[grid](
        x, bias, out, n, C,
        float(min_value), float(1.0 / divisor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


def fused_clamp_div(x, min_value, divisor):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 2048
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_clamp_div_kernel[grid](
        x, out, n,
        float(min_value), float(1.0 / divisor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        # Disable bias in conv so we can fuse it into the epilogue
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        # Create a bias parameter matching the original behavior
        self.bias = nn.Parameter(torch.empty(out_channels))
        # Initialize the same way nn.ConvTranspose3d would initialize its bias
        import math
        fan_in = in_channels * (kernel_size ** 3)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

        # Convert weight to channels_last_3d for faster cuDNN algos
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(
            memory_format=torch.channels_last_3d
        )

        self.min_value = float(min_value)
        self.divisor = float(divisor)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        bias = self.bias.contiguous()
        x = fused_bias_clamp_div_cl(x, bias, self.min_value, self.divisor)
        return x