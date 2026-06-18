import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, GROUP_SIZE, NUM_GROUPS,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (batch, group)
    pid = tl.program_id(0)
    n = pid // NUM_GROUPS
    g = pid % NUM_GROUPS

    # number of elements in this group = GROUP_SIZE * HW
    group_elems = GROUP_SIZE * HW
    
    # base offset for this (n, g)
    base = n * C * HW + g * GROUP_SIZE * HW

    # pass 1: compute sum and sum of squares of GELU(x)
    sum_val = 0.0
    sum_sq = 0.0
    
    for off in tl.range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        gelu = tl.where(mask, gelu, 0.0)
        sum_val += tl.sum(gelu, axis=0)
        sum_sq += tl.sum(gelu * gelu, axis=0)

    mean = sum_val / group_elems
    var = sum_sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # pass 2: normalize and apply affine
    for off in tl.range(0, group_elems, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        gelu = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
        
        # channel index within the group: idx // HW
        c_in_group = idx // HW
        c_global = g * GROUP_SIZE + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        
        out = (gelu - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, out, mask=mask)


def gelu_groupnorm(x, weight, bias, num_groups, eps=1e-5):
    assert x.is_cuda and x.is_contiguous()
    N, C, H, W = x.shape
    HW = H * W
    GROUP_SIZE = C // num_groups
    y = torch.empty_like(x)
    
    grid = (N * num_groups,)
    BLOCK = 1024
    _gelu_groupnorm_kernel[grid](
        x, y, weight, bias,
        N, C, HW, GROUP_SIZE, num_groups,
        eps,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=2,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        y = gelu_groupnorm(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.num_groups,
            self.group_norm.eps,
        )
        return y