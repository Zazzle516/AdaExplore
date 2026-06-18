import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_tanh_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    total_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_elements

    # compute channel index for bias
    c_idx = (offs // HW) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    # tanh
    e2 = tl.exp(2.0 * y)
    y = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + offs, y, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    total = N * C * HW
    out = torch.empty_like(x)
    BLOCK = 4096
    grid = (triton.cdiv(total, BLOCK),)
    fused_bias_tanh_kernel[grid](
        x, bias, out,
        N, C, HW,
        total,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        return fused_bias_tanh(x, bias_flat)