import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 256}, num_warps=8, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def _epilogue_kernel_cl(
    in_ptr, bias_ptr, out_ptr,
    NHW, C,
    constant_value, scaling_factor,
    BLOCK_C: tl.constexpr,
):
    # channels_last: layout is (N, H, W, C) contiguous; pid_0 is spatial-batch index
    pid_s = tl.program_id(0)  # over N*H*W
    pid_c = tl.program_id(1)  # over C tiles

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = pid_s * C + offs_c

    x = tl.load(in_ptr + base, mask=mask_c, other=0.0)
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)

    x = tl.minimum(x, constant_value)
    x = x + b
    x = x * scaling_factor

    tl.store(out_ptr + base, x, mask=mask_c)


def fused_epilogue(conv_out, bias, constant_value, scaling_factor):
    # conv_out is channels_last: (N, C, H, W) with stride (C*H*W, 1, W*C, C)
    N, C, H, W = conv_out.shape
    NHW = N * H * W
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(conv_out, memory_format=torch.channels_last)

    grid = lambda meta: (NHW, triton.cdiv(C, meta['BLOCK_C']))
    _epilogue_kernel_cl[grid](
        conv_out, bias_flat, out,
        NHW, C,
        float(constant_value), float(scaling_factor),
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

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        y = self.conv(x)
        return fused_epilogue(y, self.bias, self.constant_value, self.scaling_factor)