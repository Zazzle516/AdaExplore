import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_tanh_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    base = (pid_n * C + pid_c) * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + pid_c)
    y = x - b
    # tanh via sigmoid trick: tanh(z) = 2*sigmoid(2z) - 1
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    tl.store(out_ptr + base + offs, y, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (triton.cdiv(HW, BLOCK), C, N)
    fused_bias_tanh_kernel[grid](
        x, bias, out,
        N, C, HW,
        BLOCK=BLOCK,
        num_warps=4,
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