import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, S,  # S = D*H*W
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index
    c = (offsets // S) % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    # ReLU
    x = tl.maximum(x, 0.0)
    # LeakyReLU (no-op since x>=0)
    # GELU (exact): 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    # Sigmoid
    s = 1.0 / (1.0 + tl.exp(-g))
    # Add bias
    out = s + b

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_act_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    total = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_activation_bias_kernel[grid](
        x, bias, out,
        N, C, S, total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        bias_flat = self.bias.view(-1).contiguous()
        x = fused_act_bias(x, bias_flat)
        return x