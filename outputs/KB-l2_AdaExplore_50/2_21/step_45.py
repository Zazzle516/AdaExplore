import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_gn_kernel(
    x_ptr,          # input: (N, C, H, W) after conv
    bias_ptr,       # (C,)
    scale_ptr,      # (C,)
    gn_weight_ptr,  # (C,)
    gn_bias_ptr,    # (C,)
    out_ptr,        # output (N, C, H, W)
    N, C, HW,
    num_groups,
    channels_per_group,
    group_size,     # channels_per_group * HW
    eps,
    BLOCK_HW: tl.constexpr,
    CPG: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // num_groups
    g = pid % num_groups

    c_start = g * CPG

    # Pass 1: compute sigmoid((x+b)*s), store to out, accumulate stats
    sum_x = 0.0
    sum_x2 = 0.0

    for ci in tl.static_range(0, CPG):
        c = c_start + ci
        b = tl.load(bias_ptr + c)
        s = tl.load(scale_ptr + c)
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            base = n * C * HW + c * HW + offs
            x = tl.load(x_ptr + base, mask=mask, other=0.0)
            v = tl.sigmoid((x + b) * s)
            v = tl.where(mask, v, 0.0)
            sum_x += tl.sum(v, axis=0)
            sum_x2 += tl.sum(v * v, axis=0)
            tl.store(out_ptr + base, v, mask=mask)

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: read sigmoid output from out, normalize, write back
    for ci in tl.static_range(0, CPG):
        c = c_start + ci
        gw = tl.load(gn_weight_ptr + c)
        gb = tl.load(gn_bias_ptr + c)
        for hw_start in range(0, HW, BLOCK_HW):
            offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = offs < HW
            base = n * C * HW + c * HW + offs
            v = tl.load(out_ptr + base, mask=mask, other=0.0)
            y = (v - mean) * rstd * gw + gb
            tl.store(out_ptr + base, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        HW = H * W
        x = x.contiguous()
        out = torch.empty_like(x)

        channels_per_group = C // self.num_groups
        group_size = channels_per_group * HW

        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()

        BLOCK_HW = 1024
        grid = (N * self.num_groups,)

        fused_bias_scale_sigmoid_gn_kernel[grid](
            x, bias_flat, scale_flat, gn_w, gn_b, out,
            N, C, HW,
            self.num_groups,
            channels_per_group,
            group_size,
            self.eps,
            BLOCK_HW=BLOCK_HW,
            CPG=channels_per_group,
            num_warps=8,
        )
        return out