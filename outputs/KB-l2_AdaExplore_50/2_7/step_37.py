import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, S,  # S = D*H*W
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # channel index (always in-range due to modulo)
    c = (offs // S) % C
    b = tl.load(bias_ptr + c, eviction_policy='evict_last')

    # After ReLU, x_pos = max(x, 0). LeakyReLU on x_pos is identity.
    # GELU then sigmoid then + bias.
    inv_sqrt2 = 0.70710678118654752440
    x_pos = tl.maximum(x, 0.0)
    y = 0.5 * x_pos * (1.0 + tl.erf(x_pos * inv_sqrt2))
    y = tl.sigmoid(y) + b

    tl.store(out_ptr + offs, y, mask=mask)


def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    total = x.numel()
    out = torch.empty_like(x)
    bias_flat = bias.contiguous().view(-1)
    BLOCK_SIZE = 8192
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_activation_bias_kernel[grid](
        x, bias_flat, out,
        N, C, S, total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_bias(x, self.bias)
        return x