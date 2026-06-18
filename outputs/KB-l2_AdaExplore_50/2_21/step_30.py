import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_kernel(
    x_ptr, out_ptr, bias_ptr, scale_ptr, gn_weight_ptr, gn_bias_ptr,
    N, C, H, W, num_groups,
    eps,
    BLOCK_SIZE: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    HW: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    base = n * C * HW + g * CHANNELS_PER_GROUP * HW

    # Preload per-channel parameters for this group
    c_offs = g * CHANNELS_PER_GROUP + tl.arange(0, CHANNELS_PER_GROUP)
    bias_g = tl.load(bias_ptr + c_offs).to(tl.float32)
    scale_g = tl.load(scale_ptr + c_offs).to(tl.float32)
    gnw_g = tl.load(gn_weight_ptr + c_offs).to(tl.float32)
    gnb_g = tl.load(gn_bias_ptr + c_offs).to(tl.float32)

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    NUM_ITERS: tl.constexpr = (GROUP_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE

    for i in tl.static_range(0, NUM_ITERS):
        off = i * BLOCK_SIZE
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        c_in_group = idx // HW

        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + g * CHANNELS_PER_GROUP + c_in_group, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + g * CHANNELS_PER_GROUP + c_in_group, mask=mask, other=0.0).to(tl.float32)

        v = tl.sigmoid((x + b) * s)
        v = tl.where(mask, v, 0.0)
        sum_val += v
        sum_sq += v * v

    total = tl.sum(sum_val, axis=0)
    total_sq = tl.sum(sum_sq, axis=0)
    gs = GROUP_SIZE.to(tl.float32)
    mean = total / gs
    var = total_sq / gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for i in tl.static_range(0, NUM_ITERS):
        off = i * BLOCK_SIZE
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < GROUP_SIZE
        c_in_group = idx // HW

        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + g * CHANNELS_PER_GROUP + c_in_group, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + g * CHANNELS_PER_GROUP + c_in_group, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(gn_weight_ptr + g * CHANNELS_PER_GROUP + c_in_group, mask=mask, other=0.0).to(tl.float32)
        bb = tl.load(gn_bias_ptr + g * CHANNELS_PER_GROUP + c_in_group, mask=mask, other=0.0).to(tl.float32)

        v = tl.sigmoid((x + b) * s)
        v = (v - mean) * rstd
        v = v * w + bb

        tl.store(out_ptr + base + idx, v, mask=mask)


def fused_op(x, bias, scale, gn_weight, gn_bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    HW = H * W
    group_size = channels_per_group * HW
    out = torch.empty_like(x)

    x_c = x.contiguous()
    bias_c = bias.contiguous().view(-1)
    scale_c = scale.contiguous().view(-1)
    gnw = gn_weight.contiguous().view(-1)
    gnb = gn_bias.contiguous().view(-1)

    BLOCK_SIZE = 4096
    grid = (N * num_groups,)
    fused_kernel[grid](
        x_c, out, bias_c, scale_c, gnw, gnb,
        N, C, H, W, num_groups, eps,
        BLOCK_SIZE=BLOCK_SIZE,
        CHANNELS_PER_GROUP=channels_per_group,
        HW=HW,
        GROUP_SIZE=group_size,
        num_warps=8,
        num_stages=3,
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