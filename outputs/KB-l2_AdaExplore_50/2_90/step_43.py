import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel(
    x_ptr,
    sum_ptr,
    out_ptr,
    C,
    spatial,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    c_idx = (offs // spatial) % C
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # LeakyReLU(0.2)
    x = tl.where(x >= 0, x, x * 0.2)
    x = x + s
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


def fused_epilogue(x, sum_flat):
    x = x.contiguous()
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    spatial = D * H * W
    total = x.numel()
    BLOCK_SIZE = 4096
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, sum_flat, out,
        C, spatial, total,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        sum_flat = self.sum_tensor.view(-1).contiguous()
        x = fused_epilogue(x, sum_flat)
        return x