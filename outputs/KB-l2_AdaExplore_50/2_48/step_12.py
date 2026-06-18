import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    C, DHW,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_b = tl.program_id(1)
    c_idx = pid_nc % C
    
    base = pid_nc * DHW
    offs = pid_b * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < DHW
    
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + c_idx)
    bias = tl.load(bias_ptr + c_idx)
    
    y = x * scale
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    y = y * bias
    y = tl.sigmoid(y)
    
    tl.store(out_ptr + base + offs, y, mask=mask)


def fused_epilogue(x, scale, bias):
    x = x.contiguous()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    out = torch.empty_like(x)
    BLOCK_SIZE = 4096
    grid = (N * C, (DHW + BLOCK_SIZE - 1) // BLOCK_SIZE)
    fused_epilogue_kernel[grid](
        x, scale.contiguous().view(-1), bias.contiguous().view(-1), out,
        C, DHW,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
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