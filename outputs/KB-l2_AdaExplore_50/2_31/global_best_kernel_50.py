import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'C'],
)
@triton.jit
def _epilogue_kernel(
    in_ptr, bias_ptr, out_ptr,
    N, C, HW, total,
    constant_value, scaling_factor,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # offs decomposition: [N, C, HW] flattened
    c_idx = (offs // HW) % C

    x = tl.load(in_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x = tl.minimum(x, constant_value)
    x = x + b
    x = x * scaling_factor

    tl.store(out_ptr + offs, x, mask=mask)


def fused_epilogue(conv_out, bias, constant_value, scaling_factor):
    conv_out = conv_out.contiguous()
    N, C, H, W = conv_out.shape
    HW = H * W
    total = N * C * HW
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(conv_out)

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK']),)
    _epilogue_kernel[grid](
        conv_out, bias_flat, out,
        N, C, HW, total,
        float(constant_value), float(scaling_factor),
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
        y = self.conv(x)
        return fused_epilogue(y, self.bias, self.constant_value, self.scaling_factor)