import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, out_ptr, n_elements,
    ADD_VALUE: tl.constexpr, SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Mish: x * tanh(softplus(x))
    x_safe = tl.where(x > 20.0, 0.0, x)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x_safe)))
    e2 = tl.exp(2.0 * sp)
    th = 1.0 - 2.0 / (e2 + 1.0)
    y = x * th + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0) * SCALE
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x, add_value, scale):
    if not x.is_contiguous(memory_format=torch.channels_last):
        x = x.contiguous(memory_format=torch.channels_last)
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 8192
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, out, n,
        ADD_VALUE=float(add_value), SCALE=float(scale),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        x = fused_epilogue(x, self.add_value, self.scale)
        return x