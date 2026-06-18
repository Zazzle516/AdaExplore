import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=3),
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
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    base = n * C * HW + g * GROUP_SIZE * HW

    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        sum_val += tl.where(mask, gelu, 0.0)
        sum_sq += tl.where(mask, gelu * gelu, 0.0)

    mean = tl.sum(sum_val) / group_elems
    mean_sq = tl.sum(sum_sq) / group_elems
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Preload weight/bias for this group (GROUP_SIZE channels)
    c_offs = tl.arange(0, GROUP_SIZE)
    c_global = g * GROUP_SIZE + c_offs
    w_group = tl.load(weight_ptr + c_global).to(tl.float32)
    b_group = tl.load(bias_ptr + c_global).to(tl.float32)

    # Precompute per-channel scale and shift: out = gelu * (rstd*w) + (b - mean*rstd*w)
    scale_group = rstd * w_group
    shift_group = b_group - mean * scale_group

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        c_in_group = idx // HW
        scale = tl.load(scale_group + c_in_group * 0 + c_in_group)  # gather via index
        # The above won't work cleanly; use tl.gather-like via pointer arithmetic on registers:
        # Instead store scale/shift to a small shared-mem-like approach using tl.load with synthetic ptr.
        # Workaround: write scale/shift to a tiny local buffer is not feasible; use direct vector indexing.
        shift = tl.load(scale_group + c_in_group * 0 + c_in_group)
        out = gelu * scale + shift
        tl.store(y_ptr + base + idx, out, mask=mask)


# The above approach for indexing scale_group with c_in_group doesn't work since scale_group is a tensor in registers.
# We need a different strategy: use a small constant-sized lookup via tl.where chain or just reload weight/bias.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=3),
    ],
    key=['C', 'HW', 'GROUP_SIZE'],
)
@triton.jit
def gelu_groupnorm_kernel_v2(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, GROUP_SIZE: tl.constexpr, NUM_GROUPS,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    base = n * C * HW + g * GROUP_SIZE * HW

    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        sum_val += tl.where(mask, gelu, 0.0)
        sum_sq += tl.where(mask, gelu * gelu, 0.0)

    mean = tl.sum(sum_val) / group_elems
    mean_sq = tl.sum(sum_sq) / group_elems
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        c_in_group = idx // HW
        c_global = g * GROUP_SIZE + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)
        out = (gelu - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def fused_gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    x = x.contiguous()
    y = torch.empty_like(x)
    grid = (N * num_groups,)
    gelu_groupnorm_kernel_v2[grid](
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