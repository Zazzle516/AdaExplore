import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    total_elements, inner_size, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU
    x = tl.maximum(x, 0.0)
    # LeakyReLU(0.01) - after ReLU values are >= 0, so leaky_relu is identity
    # but we apply for fidelity (it's a no-op for x >= 0)
    # x = tl.where(x >= 0, x, 0.01 * x)  # same as x since x >= 0
    
    # GELU - exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    
    # Sigmoid
    x = tl.sigmoid(x)
    
    # Add bias - bias shape (C, 1, 1, 1), broadcast along inner dims
    # offset -> n, c, d, h, w; we need c
    c_idx = (offsets // inner_size) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    inner_size = D * H * W
    total = x.numel()
    out = torch.empty_like(x)
    
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    bias_flat = bias.contiguous().view(-1)
    
    fused_activation_bias_kernel[grid](
        x, bias_flat, out,
        total, inner_size, C,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
    
    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_bias(x, self.bias)
        return x