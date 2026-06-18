import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (batch, group)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_elems = CPG * HW
    base = n * C * HW + g * CPG * HW

    # Pass 1: compute sum and sum of squares of gelu(x)
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865475

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gx_masked = tl.where(mask, gx, 0.0)
        sum_val += gx_masked
        sumsq_val += gx_masked * gx_masked

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    mean = s / group_elems
    var = sq / group_elems - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize and apply affine, write output
    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        # channel index within group
        c_in_group = idx // HW
        c_global = g * CPG + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)

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
        # choose BLOCK_SIZE
        if group_elems >= 32768:
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

        grid = (N * G,)
        gelu_groupnorm_kernel[grid](
            x, y,
            self.group_norm.weight, self.group_norm.bias,
            N, C, HW, G, CPG,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=2,
        )
        return y