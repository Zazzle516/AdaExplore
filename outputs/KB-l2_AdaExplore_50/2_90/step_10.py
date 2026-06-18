import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel(
    x_ptr, sum_ptr, out_ptr,
    N, C, D, H, W,
    NEG_SLOPE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * D * H * W
    mask = offs < total

    spatial = D * H * W
    c_idx = (offs // spatial) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(sum_ptr + c_idx, mask=mask, other=0.0)

    # leaky relu
    x = tl.where(x >= 0, x, x * NEG_SLOPE)
    # add sum tensor
    x = x + s
    # clamp [-1, 1]
    x = tl.maximum(x, -1.0)
    x = tl.minimum(x, 1.0)
    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    x = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

    tl.store(out_ptr + offs, x, mask=mask)


def fused_epilogue(x, sum_tensor, neg_slope=0.2):
    x = x.contiguous()
    sum_flat = sum_tensor.contiguous().view(-1)
    out = torch.empty_like(x)
    N, C, D, H, W = x.shape
    total = x.numel()
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    fused_epilogue_kernel[grid](
        x, sum_flat, out,
        N, C, D, H, W,
        NEG_SLOPE=neg_slope,
        BLOCK=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.sum_tensor, neg_slope=0.2)
        return x