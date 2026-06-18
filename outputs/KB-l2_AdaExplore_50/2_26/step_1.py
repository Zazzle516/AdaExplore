import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_add_hardswish_kernel(
    x_ptr, add_ptr, bias_ptr, out_ptr,
    n_elements, C, spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(add_ptr + offsets, mask=mask, other=0.0)

    c_idx = (offsets // spatial) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    v = x + b + a
    # hardswish(v) = v * relu6(v+3)/6
    t = v + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    hs = v * t * (1.0 / 6.0)
    out = v * hs

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bias_add_hardswish(x, add_input, bias):
    x = x.contiguous()
    add_input = add_input.contiguous()
    bias_flat = bias.contiguous().view(-1)

    N, C, D, H, W = x.shape
    spatial = D * H * W
    n_elements = x.numel()

    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bias_add_hardswish_kernel[grid](
        x, add_input, bias_flat, out,
        n_elements, C, spatial,
        BLOCK_SIZE=BLOCK_SIZE,
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

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        return fused_bias_add_hardswish(x, add_input, self.bias)