import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_tanh_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H * W
    mask = offsets < total
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # bias is shape (C, 1, 1), broadcast over N, H, W
    # offset = n*C*H*W + c*H*W + h*W + w
    # c = (offset // (H*W)) % C
    HW = H * W
    c_idx = (offsets // HW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    y = x - b
    # tanh via exp
    y = (tl.exp(y) - tl.exp(-y)) / (tl.exp(y) + tl.exp(-y))
    
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    total = N * C * H * W
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_bias_tanh_kernel[grid](
        x, bias, out,
        N, C, H, W,
        BLOCK_SIZE=BLOCK_SIZE,
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

    def forward(self, x):
        x = self.conv_transpose(x)
        bias_flat = self.bias.view(-1).contiguous()
        x = fused_bias_tanh(x.contiguous(), bias_flat)
        return x