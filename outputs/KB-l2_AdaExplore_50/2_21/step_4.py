import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_groupnorm_kernel(
    x_ptr, out_ptr, scratch_ptr, bias_ptr, scale_ptr, gn_weight_ptr, gn_bias_ptr,
    N, C, H, W, num_groups, eps,
    BLOCK_HW: tl.constexpr,
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

    # Load per-channel constants once (CHANNELS_PER_GROUP is small, e.g. 4)
    c_offs = g * CHANNELS_PER_GROUP + tl.arange(0, CHANNELS_PER_GROUP)
    bias_vec = tl.load(bias_ptr + c_offs).to(tl.float32)   # [C_per_g]
    scale_vec = tl.load(scale_ptr + c_offs).to(tl.float32)
    gnw_vec = tl.load(gn_weight_ptr + c_offs).to(tl.float32)
    gnb_vec = tl.load(gn_bias_ptr + c_offs).to(tl.float32)

    # First pass: compute sigmoid((x+b)*s), accumulate sum and sumsq, write to scratch
    sum_acc = tl.zeros([CHANNELS_PER_GROUP, BLOCK_HW], dtype=tl.float32)
    sumsq_acc = tl.zeros([CHANNELS_PER_GROUP, BLOCK_HW], dtype=tl.float32)

    for off in range(0, HW, BLOCK_HW):
        hw_idx = off + tl.arange(0, BLOCK_HW)              # [BLOCK_HW]
        mask_hw = hw_idx < HW                              # [BLOCK_HW]
        # 2D index: [C_per_g, BLOCK_HW]
        idx2d = (tl.arange(0, CHANNELS_PER_GROUP)[:, None] * HW) + hw_idx[None, :]
        mask2d = mask_hw[None, :]
        x = tl.load(x_ptr + base + idx2d, mask=mask2d, other=0.0).to(tl.float32)
        v = (x + bias_vec[:, None]) * scale_vec[:, None]
        v = tl.sigmoid(v)
        v = tl.where(mask2d, v, 0.0)
        tl.store(scratch_ptr + base + idx2d, v, mask=mask2d)
        sum_acc += v
        sumsq_acc += v * v

    total = tl.sum(tl.sum(sum_acc, axis=1), axis=0)
    total_sq = tl.sum(tl.sum(sumsq_acc, axis=1), axis=0)
    gs = group_size.to(tl.float32)
    mean = total / gs
    var = total_sq / gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Precompute per-channel affine scale/shift:
    # out = ((v - mean) * rstd) * gnw + gnb = v * (rstd*gnw) + (gnb - mean*rstd*gnw)
    a_vec = rstd * gnw_vec
    b_vec = gnb_vec - mean * a_vec

    # Second pass: read from scratch, apply affine
    for off in range(0, HW, BLOCK_HW):
        hw_idx = off + tl.arange(0, BLOCK_HW)
        mask_hw = hw_idx < HW
        idx2d = (tl.arange(0, CHANNELS_PER_GROUP)[:, None] * HW) + hw_idx[None, :]
        mask2d = mask_hw[None, :]
        v = tl.load(scratch_ptr + base + idx2d, mask=mask2d, other=0.0).to(tl.float32)
        out = v * a_vec[:, None] + b_vec[:, None]
        tl.store(out_ptr + base + idx2d, out, mask=mask2d)


def fused_op(x, bias, scale, gn_weight, gn_bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    out = torch.empty_like(x)
    scratch = torch.empty_like(x)

    x_c = x.contiguous()
    bias_c = bias.contiguous().view(-1)
    scale_c = scale.contiguous().view(-1)
    gnw = gn_weight.contiguous().view(-1)
    gnb = gn_bias.contiguous().view(-1)

    grid = (N * num_groups,)
    BLOCK_HW = 512
    fused_bias_scale_sigmoid_groupnorm_kernel[grid](
        x_c, out, scratch, bias_c, scale_c, gnw, gnb,
        N, C, H, W, num_groups, eps,
        BLOCK_HW=BLOCK_HW,
        CHANNELS_PER_GROUP=channels_per_group,
        num_warps=8,
        num_stages=2,
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