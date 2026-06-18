import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
    ],
    key=['C', 'HW', 'GROUP_SIZE'],
)
@triton.jit
def gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, GROUP_SIZE: tl.constexpr, NUM_GROUPS,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (batch, group)
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    # base pointer for this (n, g)
    base = n * C * HW + g * GROUP_SIZE * HW
    group_elems = GROUP_SIZE * HW

    # First pass: compute mean and var of GELU(x) over the group
    # Loop channels then HW so we walk contiguous memory
    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for c in range(0, GROUP_SIZE):
        c_base = base + c * HW
        for off in range(0, HW, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < HW
            x = tl.load(x_ptr + c_base + idx, mask=mask, other=0.0).to(tl.float32)
            gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
            sum_val += tl.where(mask, gelu, 0.0)
            sum_sq += tl.where(mask, gelu * gelu, 0.0)

    mean = tl.sum(sum_val) / group_elems
    mean_sq = tl.sum(sum_sq) / group_elems
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Second pass: load w/b once per channel
    for c in range(0, GROUP_SIZE):
        c_global = g * GROUP_SIZE + c
        c_base = base + c * HW
        w = tl.load(weight_ptr + c_global)
        b = tl.load(bias_ptr + c_global)
        scale = rstd * w
        shift = b - mean * scale
        for off in range(0, HW, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < HW
            x = tl.load(x_ptr + c_base + idx, mask=mask, other=0.0).to(tl.float32)
            gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
            out = gelu * scale + shift
            tl.store(y_ptr + c_base + idx, out, mask=mask)


def fused_gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    x = x.contiguous()
    y = torch.empty_like(x)
    grid = (N * num_groups,)
    gelu_groupnorm_kernel[grid](
        x, y, weight, bias,
        N, C, HW, GROUP_SIZE, num_groups,
        eps,
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
        x = fused_gelu_groupnorm(x, self.group_norm.weight, self.group_norm.bias,
                                  self.num_groups, self.eps)
        return x