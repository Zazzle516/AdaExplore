import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gelu_groupnorm_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    N, C, HW, GROUP_SIZE, NUM_GROUPS,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    group_elems = GROUP_SIZE * HW
    base = n * C * HW + g * GROUP_SIZE * HW

    sum_x = 0.0
    sum_x2 = 0.0
    inv_sqrt2 = 0.7071067811865475

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

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def groupnorm_apply_kernel(
    x_ptr, y_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
    N, C, HW, GROUP_SIZE, NUM_GROUPS,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (batch*channel, hw-tile)
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C
    g = c // GROUP_SIZE

    mean = tl.load(mean_ptr + n * NUM_GROUPS + g)
    rstd = tl.load(rstd_ptr + n * NUM_GROUPS + g)
    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)

    scale = rstd * w
    shift = b - mean * scale
    inv_sqrt2 = 0.7071067811865475

    base = n * C * HW + c * HW
    off = pid_hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < HW
    x = tl.load(x_ptr + base + off, mask=mask, other=0.0)
    gelu = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    y = gelu * scale + shift
    tl.store(y_ptr + base + off, y, mask=mask)


def fused_gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups

    y = torch.empty_like(x)
    mean = torch.empty((N * num_groups,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N * num_groups,), device=x.device, dtype=torch.float32)

    grid1 = (N * num_groups,)
    BLOCK1 = 4096
    gelu_groupnorm_stats_kernel[grid1](
        x, mean, rstd,
        N, C, HW, GROUP_SIZE, num_groups,
        eps,
        BLOCK_SIZE=BLOCK1,
        num_warps=16,
        num_stages=2,
    )

    BLOCK2 = 2048
    grid2 = (N * C, (HW + BLOCK2 - 1) // BLOCK2)
    groupnorm_apply_kernel[grid2](
        x, y, mean, rstd, weight, bias,
        N, C, HW, GROUP_SIZE, num_groups,
        BLOCK_SIZE=BLOCK2,
        num_warps=8,
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