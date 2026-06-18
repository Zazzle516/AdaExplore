import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel(
    x_ptr,       # conv output [N, C, D, H, W]
    sum_ptr,     # sum tensor [C]
    out_ptr,     # output
    C,
    SPATIAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)

    c = pid_nc % C
    base = pid_nc * SPATIAL
    offs = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < SPATIAL
    addrs = base + offs

    x = tl.load(x_ptr + addrs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c)

    # LeakyReLU(0.2)
    x = tl.where(x > 0, x, x * 0.2)
    # Add
    x = x + s
    # Clamp
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + addrs, x, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        x = x.contiguous()
        sum_flat = self.sum_tensor.contiguous().view(-1)
        out = torch.empty_like(x)
        spatial = D * H * W
        BLOCK_SIZE = 4096
        grid = (N * C, (spatial + BLOCK_SIZE - 1) // BLOCK_SIZE)
        fused_epilogue_kernel[grid](
            x, sum_flat, out,
            C,
            SPATIAL=spatial,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
        return out