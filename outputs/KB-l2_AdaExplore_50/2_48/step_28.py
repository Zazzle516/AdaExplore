import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    total, DHW, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    c_idx = (offsets // DHW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    y = x * scale
    # tanh
    e = tl.exp(-2.0 * y)
    y = (1.0 - e) / (1.0 + e)
    y = y * bias
    y = tl.sigmoid(y)
    
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x, scale, bias):
    x = x.contiguous()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    out = torch.empty_like(x)
    total = x.numel()
    BLOCK_SIZE = 2048
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, scale.contiguous().view(-1), bias.contiguous().view(-1), out,
        total, DHW, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.scaling_factor, self.bias)
        return x