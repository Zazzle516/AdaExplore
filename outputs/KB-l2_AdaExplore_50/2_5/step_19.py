import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bias_tanh_nhwc_kernel(
    x_ptr, b_ptr, out_ptr,
    TOTAL, C,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    c_idx = offs % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    e2x = tl.exp(2.0 * y)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _bias_tanh_nchw_kernel(
    x_ptr, b_ptr, out_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    base = pid_n * C * HW + pid_c * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + pid_c)
    y = x - b
    e2x = tl.exp(2.0 * y)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + base + offs, out, mask=mask)


def fused_bias_tanh_nhwc(x, bias):
    # x is in channels_last memory format (NCHW logical, NHWC physical)
    N, C, H, W = x.shape
    TOTAL = N * C * H * W
    out = torch.empty_like(x, memory_format=torch.channels_last)
    BLOCK = 4096
    grid = ((TOTAL + BLOCK - 1) // BLOCK,)
    _bias_tanh_nhwc_kernel[grid](x, bias, out, TOTAL, C, BLOCK=BLOCK, num_warps=8)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        # Convert weights to channels_last for faster cuDNN path
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        b = self.bias.view(-1).contiguous()
        if x.is_contiguous(memory_format=torch.channels_last):
            return fused_bias_tanh_nhwc(x, b)
        else:
            x = x.contiguous()
            N, C, H, W = x.shape
            HW = H * W
            out = torch.empty_like(x)
            BLOCK = 1024
            grid = ((HW + BLOCK - 1) // BLOCK, C, N)
            _bias_tanh_nchw_kernel[grid](x, b, out, N, C, HW, BLOCK=BLOCK, num_warps=4)
            return out