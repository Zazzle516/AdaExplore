import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


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


def fused_clamp_div(x, min_value, divisor):
    if not x.is_contiguous(memory_format=torch.channels_last_3d) and not x.is_contiguous():
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
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Use channels_last_3d memory format for potentially faster cuDNN algorithm
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        if x.dim() == 5:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = fused_clamp_div(x, self.min_value, self.divisor)
        return x