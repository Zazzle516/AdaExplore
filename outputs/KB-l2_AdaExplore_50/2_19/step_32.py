import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    HW, CPG, group_elems,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    G = tl.num_programs(1)
    C = G * CPG

    base = pid * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865475

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gx_masked = tl.where(mask, gx, 0.0)
        sum_val += gx_masked
        sumsq_val += gx_masked * gx_masked

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    inv_n = 1.0 / group_elems
    mean = s * inv_n
    var = sq * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    # preload affine for this group: CPG channels starting at g*CPG
    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        c_in_group = idx // HW
        c_global = g * CPG + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)

        y = (gx - mean) * rstd * w + b
        tl.store(y_ptr + base + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, H, W = x.shape
        HW = H * W
        G = self.num_groups
        CPG = C // G

        x = x.contiguous()
        y = torch.empty_like(x)

        group_elems = CPG * HW

        if group_elems >= 16384:
            BLOCK_SIZE = 2048
            num_warps = 8
        elif group_elems >= 4096:
            BLOCK_SIZE = 1024
            num_warps = 8
        elif group_elems >= 1024:
            BLOCK_SIZE = 512
            num_warps = 4
        else:
            BLOCK_SIZE = 256
            num_warps = 4

        grid = (N, G)
        fused_gelu_groupnorm_kernel[grid](
            x, y,
            self.group_norm.weight, self.group_norm.bias,
            HW, CPG, group_elems,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=3,
        )
        return y