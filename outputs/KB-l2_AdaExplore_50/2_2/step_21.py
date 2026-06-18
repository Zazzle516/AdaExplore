import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, HW,
    scaling_factor,
    inv_scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)  # index into N*C
    tile = tl.program_id(1)
    c = row % C
    b = tl.load(bias_ptr + c)

    offs = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < HW

    base = row * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)

    y = x + b
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = y * scaling_factor
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = y * inv_scaling_factor

    tl.store(out_ptr + base + offs, y, mask=mask)


def fused_epilogue(x, bias, scaling_factor):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    HW = H * W
    NC = N * C
    grid = lambda META: (NC, (HW + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'])
    _epilogue_kernel[grid](
        x, bias, out,
        C, HW,
        float(scaling_factor),
        1.0 / float(scaling_factor),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        bias_flat = self.bias.view(-1).contiguous().to(x.dtype)
        x = fused_epilogue(x, bias_flat, self.scaling_factor)
        return x