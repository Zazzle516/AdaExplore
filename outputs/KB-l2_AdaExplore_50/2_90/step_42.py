import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel_cl(
    x_ptr,
    sum_ptr,
    out_ptr,
    C,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    # channels_last layout: channel is innermost contiguous dim
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    c_idx = offs % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)

    x = tl.where(x >= 0, x, x * 0.2)
    x = x + s
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


@triton.jit
def fused_epilogue_kernel_contig(
    x_ptr,
    sum_ptr,
    out_ptr,
    C,
    spatial,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    c_idx = (offs // spatial) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)

    x = tl.where(x >= 0, x, x * 0.2)
    x = x + s
    x = tl.minimum(tl.maximum(x, -1.0), 1.0)
    inv_sqrt2 = 0.7071067811865475
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


def fused_epilogue(x, sum_flat, channels_last=False):
    N, C, D, H, W = x.shape
    total = x.numel()
    BLOCK_SIZE = 8192
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    if channels_last:
        out = torch.empty_like(x, memory_format=torch.channels_last_3d)
        fused_epilogue_kernel_cl[grid](
            x, sum_flat, out,
            C, total,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=2,
        )
    else:
        x = x.contiguous()
        out = torch.empty_like(x)
        spatial = D * H * W
        fused_epilogue_kernel_contig[grid](
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
        self.conv = self.conv.to(memory_format=torch.channels_last_3d)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv(x)
        cl = x.is_contiguous(memory_format=torch.channels_last_3d) and not x.is_contiguous()
        sum_flat = self.sum_tensor.contiguous().view(-1)
        x = fused_epilogue(x, sum_flat, channels_last=cl)
        return x