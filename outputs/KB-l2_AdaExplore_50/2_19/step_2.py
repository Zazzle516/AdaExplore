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
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        gelu = tl.where(mask, gelu, 0.0)
        sum_val += gelu
        sum_sq += gelu * gelu

    inv_n = 1.0 / group_elems
    mean = tl.sum(sum_val) * inv_n
    mean_sq = tl.sum(sum_sq) * inv_n
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Preload affine for this group (small: GROUP_SIZE channels)
    c_offs = g * GROUP_SIZE + tl.arange(0, 16)  # GROUP_SIZE assumed <=16, mask
    cmask = tl.arange(0, 16) < GROUP_SIZE
    w_g = tl.load(weight_ptr + c_offs, mask=cmask, other=0.0)
    b_g = tl.load(bias_ptr + c_offs, mask=cmask, other=0.0)

    neg_mean_rstd = -mean * rstd

    for off in range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        c_in_group = idx // HW
        # gather w,b from preloaded vector
        # Use tl.where chain since GROUP_SIZE is small? Use load via pointer
        w = tl.load(weight_ptr + g * GROUP_SIZE + c_in_group, mask=mask, other=0.0)
        b = tl.load(bias_ptr + g * GROUP_SIZE + c_in_group, mask=mask, other=0.0)
        out = (gelu * rstd + neg_mean_rstd) * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def fused_gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    x = x.contiguous()
    y = torch.empty_like(x)
    grid = (N * num_groups,)
    BLOCK = 2048
    gelu_groupnorm_kernel[grid](
        x, y, weight, bias,
        HW, GROUP_SIZE, num_groups, C,
        eps,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=3,
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