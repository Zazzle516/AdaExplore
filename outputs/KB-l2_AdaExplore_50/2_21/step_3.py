import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_groupnorm_kernel(
    x_ptr, out_ptr, bias_ptr, scale_ptr, gn_weight_ptr, gn_bias_ptr,
    N, C, H, W, num_groups, eps,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
):
    # one program per (batch, group)
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    HW = H * W
    group_size = CHANNELS_PER_GROUP * HW

    # base offset into x for this (n, g)
    base = n * C * HW + g * CHANNELS_PER_GROUP * HW

    # First pass: compute mean and var of sigmoid((x+bias)*scale) over the group
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        # channel index within group
        c_in_group = idx // HW
        c_global = g * CHANNELS_PER_GROUP + c_in_group

        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + c_global, mask=mask, other=0.0).to(tl.float32)

        v = (x + b) * s
        v = tl.sigmoid(v)
        v = tl.where(mask, v, 0.0)
        sum_val += v
        sum_sq += v * v

    total = tl.sum(sum_val, axis=0)
    total_sq = tl.sum(sum_sq, axis=0)
    gs = group_size.to(tl.float32)
    mean = total / gs
    var = total_sq / gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        c_in_group = idx // HW
        c_global = g * CHANNELS_PER_GROUP + c_in_group

        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + c_global, mask=mask, other=0.0).to(tl.float32)

        v = (x + b) * s
        v = tl.sigmoid(v)
        v = (v - mean) * rstd

        w = tl.load(gn_weight_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        bb = tl.load(gn_bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        v = v * w + bb

        tl.store(out_ptr + base + idx, v, mask=mask)


def fused_op(x, bias, scale, gn_weight, gn_bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    out = torch.empty_like(x)

    x_c = x.contiguous()
    bias_c = bias.contiguous().view(-1)
    scale_c = scale.contiguous().view(-1)
    gnw = gn_weight.contiguous().view(-1)
    gnb = gn_bias.contiguous().view(-1)

    grid = (N * num_groups,)
    BLOCK_SIZE = 1024
    fused_bias_scale_sigmoid_groupnorm_kernel[grid](
        x_c, out, bias_c, scale_c, gnw, gnb,
        N, C, H, W, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_GROUP=channels_per_group,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        x = fused_op(x, self.bias, self.scale, self.group_norm.weight, self.group_norm.bias,
                     self.num_groups, self.eps)
        return x