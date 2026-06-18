import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, GROUP_SIZE, NUM_GROUPS,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (batch, group)
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW  # number of elements per group
    base = n * C * HW + g * GROUP_SIZE * HW

    inv_sqrt2 = 0.7071067811865475

    # First pass: compute GELU on the fly, accumulate sums
    sum_x = 0.0
    sum_x2 = 0.0

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gelu_m = tl.where(mask, gelu, 0.0)
        sum_x += tl.sum(gelu_m)
        sum_x2 += tl.sum(gelu_m * gelu_m)

    mean = sum_x / group_elems
    var = sum_x2 / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: recompute GELU from x, normalize, apply affine
    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        c_local = idx // HW
        c_global = g * GROUP_SIZE + c_local
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)

        out = (gelu - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def fused_gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    y = torch.empty_like(x)

    grid = (N * num_groups,)
    group_elems = GROUP_SIZE * HW
    if group_elems >= 65536:
        BLOCK_SIZE = 8192
        num_warps = 16
    elif group_elems >= 8192:
        BLOCK_SIZE = 4096
        num_warps = 8
    elif group_elems >= 4096:
        BLOCK_SIZE = 2048
        num_warps = 8
    elif group_elems >= 1024:
        BLOCK_SIZE = 1024
        num_warps = 4
    else:
        BLOCK_SIZE = 256
        num_warps = 2

    gelu_groupnorm_kernel[grid](
        x, y, weight, bias,
        N, C, HW, GROUP_SIZE, num_groups,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        y = fused_gelu_groupnorm(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.eps,
        )
        return y