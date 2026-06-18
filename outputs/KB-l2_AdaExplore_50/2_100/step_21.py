import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def fused_clamp_div_kernel(
    x_ptr, out_ptr, n_elements,
    MIN_VAL: tl.constexpr, INV_DIV: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.maximum(x, MIN_VAL)
    x = x * INV_DIV
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_clamp_div(x: torch.Tensor, min_value: float, divisor: float):
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    n = x_c.numel()
    BLOCK_SIZE = 8192
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_clamp_div_kernel[grid](
        x_c, out, n,
        MIN_VAL=float(min_value), INV_DIV=float(1.0 / divisor),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out.view_as(x)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, min_value, divisor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Move weights to channels_last_3d for faster cuDNN algorithms
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        self.min_value = min_value
        self.divisor = divisor

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        x = fused_clamp_div(x, self.min_value, self.divisor)
        return x