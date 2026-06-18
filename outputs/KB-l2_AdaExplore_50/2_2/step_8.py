import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    scaling_factor,
    inv_scaling,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # compute channel index
    hw = H * W
    c = (offsets // hw) % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    y = x + b
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = y * scaling_factor
    y = tl.minimum(tl.maximum(y, 0.0), 1.0)
    y = y * inv_scaling

    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x, bias, scaling_factor):
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, bias, out,
        N, C, H, W,
        scaling_factor,
        1.0 / scaling_factor,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
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