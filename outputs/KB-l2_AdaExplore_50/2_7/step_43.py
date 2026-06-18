import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
    ],
    key=['inner_size'],
)
@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    inner_size,
    BLOCK_SIZE: tl.constexpr,
):
    nc = tl.program_id(0)
    blk = tl.program_id(1)
    
    # Per-program channel scalar bias load
    c = nc % tl.num_programs(0)  # not used, we get c from nc directly
    # We will pass C via grid; use nc directly: bias channel = nc % C, but
    # since program_id(0) iterates N*C and bias is per-channel of size C,
    # the channel is nc modulo C. We'll compute it from a dedicated arg.
    
    base = nc * inner_size + blk * BLOCK_SIZE
    offsets = base + tl.arange(0, BLOCK_SIZE)
    inner_offsets = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = inner_offsets < inner_size
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU + LeakyReLU (identity for non-negative)
    x = tl.maximum(x, 0.0)
    
    # GELU - exact
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    
    # Sigmoid
    x = tl.sigmoid(x)
    
    tl.store(out_ptr + offsets, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
    ],
    key=['inner_size'],
)
@triton.jit
def fused_activation_bias_kernel_v2(
    x_ptr, bias_ptr, out_ptr,
    total_elements, inner_size, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU + LeakyReLU (identity for non-negative)
    x = tl.maximum(x, 0.0)
    
    # GELU - exact
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    
    # Sigmoid
    x = tl.sigmoid(x)
    
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
    
    bias_flat = bias.contiguous().view(-1)
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    
    fused_activation_bias_kernel_v2[grid](
        x, bias_flat, out,
        total, inner_size, C,
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