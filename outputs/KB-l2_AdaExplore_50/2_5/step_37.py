import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_tanh_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, HW,
    BLOCK: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    c_idx = pid_nc % C
    b = tl.load(bias_ptr + c_idx)

    offs = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    base = pid_nc * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    y = x - b
    y = tl.extra.cuda.libdevice.tanh(y)
    tl.store(out_ptr + base + offs, y, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK = 4096
    grid = (triton.cdiv(HW, BLOCK), N * C)
    fused_bias_tanh_kernel[grid](
        x, bias, out,
        C, HW,
        BLOCK=BLOCK,
        num_warps=4,
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
        bias_flat = self.bias.view(-1)
        return fused_bias_tanh(x, bias_flat)