import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    constant_value, scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    HW = H * W
    CHW = C * HW
    c_idx = (offs // HW) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x = tl.minimum(x, constant_value)
    x = (x + b) * scaling_factor

    tl.store(out_ptr + offs, x, mask=mask)


def fused_epilogue(x, bias, constant_value, scaling_factor):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    bias_flat = bias.contiguous().view(-1)
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _epilogue_kernel[grid](
        x, bias_flat, out,
        N, C, H, W,
        float(constant_value), float(scaling_factor),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.bias, self.constant_value, self.scaling_factor)
        return x