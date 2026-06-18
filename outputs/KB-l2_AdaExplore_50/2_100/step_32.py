import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_clamp_div_kernel(
    x_ptr, bias_ptr, out_ptr, n_elements,
    C, DHW,
    min_value, inv_divisor,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // DHW) % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x = x + b
    x = tl.maximum(x, min_value)
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
    x = tl.maximum(x, min_value)
    x = x * inv_divisor
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_bias_clamp_div(x: torch.Tensor, bias: torch.Tensor, min_value: float, divisor: float):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bias_clamp_div_kernel[grid](
        x, bias, out, n,
        C, DHW,
        float(min_value), float(1.0 / divisor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


def fused_clamp_div(x: torch.Tensor, min_value: float, divisor: float):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_clamp_div_kernel[grid](
        x, out, n,
        float(min_value), float(1.0 / divisor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        # Build conv_transpose without bias; keep bias as a separate parameter
        # so we can fuse the bias-add into the clamp/div epilogue.
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        # Initialize bias the same way nn.ConvTranspose3d does.
        ref = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                 stride=stride, padding=padding, bias=True)
        self.bias = nn.Parameter(ref.bias.detach().clone())
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last_3d)
        # Make sure conv weight is also channels_last_3d for cuDNN.
        if self.conv_transpose.weight.is_contiguous(memory_format=torch.channels_last_3d) is False:
            with torch.no_grad():
                self.conv_transpose.weight.data = self.conv_transpose.weight.data.contiguous(
                    memory_format=torch.channels_last_3d
                )
        x = self.conv_transpose(x)
        x = fused_bias_clamp_div(x, self.bias, self.min_value, self.divisor)
        return x