import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel_cl(
    x_ptr, bias_ptr, out_ptr,
    C,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    y = (2.0 * x + b) * x + x
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def fused_kernel_nc(
    x_ptr, bias_ptr, out_ptr,
    C, S,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // S) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    y = (2.0 * x + b) * x + x
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_op(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    N, C, D, H, W = x.shape
    total = x.numel()
    bias_flat = bias.contiguous().view(-1)
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    is_cl = x.is_contiguous(memory_format=torch.channels_last_3d)
    if is_cl:
        out = torch.empty_like(x, memory_format=torch.channels_last_3d)
        fused_kernel_cl[grid](
            x, bias_flat, out,
            C, total,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    else:
        x = x.contiguous()
        out = torch.empty_like(x)
        S = D * H * W
        fused_kernel_nc[grid](
            x, bias_flat, out,
            C, S, total,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Try channels_last_3d for conv weights to enable faster cuDNN paths on Ada
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        if x.dim() == 5:
            x = x.contiguous(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        return fused_op(x, self.bias)