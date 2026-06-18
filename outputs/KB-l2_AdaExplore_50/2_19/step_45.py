import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    HW, GROUP_SIZE, NUM_GROUPS, C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    base = n * C * HW + g * GROUP_SIZE * HW

    inv_sqrt2 = 0.7071067811865475

    sum_x = 0.0
    sum_x2 = 0.0

    # First pass: compute statistics over GELU(x)
    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gelu = tl.where(mask, gelu, 0.0)
        sum_x += tl.sum(gelu)
        sum_x2 += tl.sum(gelu * gelu)

    inv_n = 1.0 / group_elems
    mean = sum_x * inv_n
    var = sum_x2 * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Preload affine for this group: GROUP_SIZE channels
    c_offs = g * GROUP_SIZE + tl.arange(0, 64)  # safe upper bound on group size
    c_mask = tl.arange(0, 64) < GROUP_SIZE
    # We will instead compute weight/bias per element via channel index

    # Second pass: normalize, affine, store
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

    BLOCK_SIZE = 4096
    num_warps = 8
    num_stages = 3

    gelu_groupnorm_kernel[grid](
        x, y, weight, bias,
        HW, GROUP_SIZE, num_groups, C,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
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