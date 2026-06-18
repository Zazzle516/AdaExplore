import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def fused_bias_tanh_nhwc_kernel(
    x_ptr, bias_ptr, out_ptr,
    total, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    e2 = tl.exp(-2.0 * tl.abs(y))
    t = (1.0 - e2) / (1.0 + e2)
    y = tl.where(y >= 0, t, -t)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def fused_bias_tanh_nchw_kernel(
    x_ptr, bias_ptr, out_ptr,
    total, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // HW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    e2 = tl.exp(-2.0 * tl.abs(y))
    t = (1.0 - e2) / (1.0 + e2)
    y = tl.where(y >= 0, t, -t)
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    out = torch.empty_like(x, memory_format=torch.channels_last if x.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format)
    total = N * C * H * W
    BLOCK_SIZE = 4096
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    if x.is_contiguous(memory_format=torch.channels_last):
        fused_bias_tanh_nhwc_kernel[grid](
            x, bias, out,
            total, C,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    else:
        HW = H * W
        fused_bias_tanh_nchw_kernel[grid](
            x, bias, out,
            total, C, HW,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=4,
            num_stages=2,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        bias_flat = self.bias.view(-1).contiguous()
        return fused_bias_tanh(x, bias_flat)