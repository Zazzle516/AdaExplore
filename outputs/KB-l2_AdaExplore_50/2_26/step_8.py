import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_add_hardswish_kernel(
    x_ptr, add_ptr, bias_ptr, out_ptr,
    n_elements, C, spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    a = tl.load(add_ptr + offsets, mask=mask, other=0.0)
    # bias: shape (C,1,1,1,1) broadcast across batch and spatial
    c_idx = (offsets // spatial) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    v = x + a + b
    # hardswish(v) = v * relu6(v+3) / 6
    relu6 = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
    hs = v * relu6 * (1.0 / 6.0)
    out = v * hs
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_add_hardswish(x, add_input, bias):
    x = x.contiguous()
    add_input = add_input.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    n = x.numel()
    N, C, D, H, W = x.shape
    spatial = D * H * W
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_add_hardswish_kernel[grid](
        x, add_input, bias_flat, out,
        n, C, spatial,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x, add_input):
        x = self.conv_transpose(x)
        # fold add_input + bias broadcast + hardswish*x into one kernel
        return fused_add_hardswish(x, add_input, self.bias)