import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, HW,
    upper_bound,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    c = (offsets // HW) % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    y = x + b
    # clamp(x+b, 0, min(1, 1/s)) -- equivalent to the original chain
    y = tl.minimum(tl.maximum(y, 0.0), upper_bound)

    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x, bias, scaling_factor):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    HW = H * W
    upper_bound = min(1.0, 1.0 / scaling_factor)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_epilogue_kernel[grid](
        x, bias, out,
        C, HW,
        upper_bound,
        n_elements,
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
        x = fused_epilogue(x, self.bias.view(-1).contiguous(), self.scaling_factor)
        return x