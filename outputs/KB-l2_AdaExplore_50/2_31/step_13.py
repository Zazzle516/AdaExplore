import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _epilogue_kernel_nchw(
    x_ptr, bias_ptr, out_ptr,
    C, HW,
    constant_value, scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c_idx = (offs // HW) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x = tl.minimum(x, constant_value)
    x = (x + b) * scaling_factor

    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def _epilogue_kernel_nhwc(
    x_ptr, bias_ptr, out_ptr,
    C,
    constant_value, scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c_idx = offs % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x = tl.minimum(x, constant_value)
    x = (x + b) * scaling_factor

    tl.store(out_ptr + offs, x, mask=mask)


def fused_epilogue(x, bias, constant_value, scaling_factor):
    N, C, H, W = x.shape
    n_elements = x.numel()
    bias_flat = bias.contiguous().view(-1)
    BLOCK_SIZE = 2048

    is_channels_last = x.is_contiguous(memory_format=torch.channels_last)

    if is_channels_last:
        out = torch.empty_like(x, memory_format=torch.channels_last)
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _epilogue_kernel_nhwc[grid](
            x, bias_flat, out,
            C,
            float(constant_value), float(scaling_factor),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
    else:
        x = x.contiguous()
        out = torch.empty_like(x)
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        _epilogue_kernel_nchw[grid](
            x, bias_flat, out,
            C, H * W,
            float(constant_value), float(scaling_factor),
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv = self.conv.to(memory_format=torch.channels_last)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        torch.backends.cudnn.benchmark = True

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv(x)
        x = fused_epilogue(x, self.bias, self.constant_value, self.scaling_factor)
        return x