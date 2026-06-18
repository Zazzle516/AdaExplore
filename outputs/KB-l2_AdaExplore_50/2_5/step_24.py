import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bias_tanh_kernel(
    x_ptr, b_ptr, out_ptr,
    HW,
    BLOCK: tl.constexpr,
):
    pid_s = tl.program_id(0)  # spatial block
    pid_c = tl.program_id(1)  # channel
    pid_n = tl.program_id(2)  # batch

    offs = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    # base computed by program_id; bias is single broadcast load
    base = (pid_n * tl.num_programs(1) + pid_c) * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + pid_c)
    y = x - b
    e2x = tl.exp(2.0 * y)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + base + offs, out, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK = 8192
    grid = ((HW + BLOCK - 1) // BLOCK, C, N)
    _bias_tanh_kernel[grid](x, bias, out, HW, BLOCK=BLOCK, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        b = self.bias.view(-1).contiguous()
        return fused_bias_tanh(x.contiguous(), b)