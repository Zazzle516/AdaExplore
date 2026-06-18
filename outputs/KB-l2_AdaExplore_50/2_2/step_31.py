import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, HW,
    inv_scale,
    scaling_factor,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c, mask=mask & (c < C), other=0.0)

    v = x + b
    v = tl.minimum(tl.maximum(v, 0.0), 1.0)
    v = v * scaling_factor
    v = tl.minimum(tl.maximum(v, 0.0), 1.0)
    v = v * inv_scale

    tl.store(out_ptr + offsets, v, mask=mask)


def fused_epilogue(x, bias, scaling_factor):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = int(x.numel())
    HW = int(H * W)
    C_int = int(C)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_epilogue_kernel[grid](
        x, bias, out,
        C_int, HW,
        1.0 / scaling_factor,
        scaling_factor,
        n_elements,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        bias_flat = self.bias.view(-1).contiguous()
        return fused_epilogue(x, bias_flat, self.scaling_factor)