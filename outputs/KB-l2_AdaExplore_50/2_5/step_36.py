import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_tanh_kernel(
    x_ptr, bias_ptr, out_ptr,
    TOTAL, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    c_idx = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    # tanh via sigmoid trick: tanh(z) = 2*sigmoid(2z) - 1
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    tl.store(out_ptr + offs, y, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    TOTAL = N * C * HW
    out = torch.empty_like(x)
    BLOCK = 2048
    grid = (triton.cdiv(TOTAL, BLOCK),)
    fused_bias_tanh_kernel[grid](
        x, bias, out,
        TOTAL, C, HW,
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
        bias_flat = self.bias.view(-1).contiguous()
        return fused_bias_tanh(x.contiguous(), bias_flat)