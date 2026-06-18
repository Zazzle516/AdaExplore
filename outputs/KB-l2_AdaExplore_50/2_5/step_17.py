import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bias_tanh_cl_kernel(
    x_ptr, b_ptr, out_ptr,
    TOTAL, C,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    # channels_last: innermost dim is C
    c_idx = offs % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    e2x = tl.exp(2.0 * y)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


def fused_bias_tanh_cl(x, bias):
    # x is in channels_last memory format (NHWC contiguous)
    N, C, H, W = x.shape
    TOTAL = N * C * H * W
    out = torch.empty_like(x)
    BLOCK = 4096
    grid = ((TOTAL + BLOCK - 1) // BLOCK,)
    _bias_tanh_cl_kernel[grid](x, bias, out, TOTAL, C, BLOCK=BLOCK, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Convert weight to channels_last for faster cuDNN path
        self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        b = self.bias.view(-1).contiguous()
        return fused_bias_tanh_cl(x, b)