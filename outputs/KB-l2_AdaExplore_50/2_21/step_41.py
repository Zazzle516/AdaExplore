import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_bias_scale_sigmoid_gn_kernel(
    x_ptr, bias_ptr, scale_ptr, gn_w_ptr, gn_b_ptr, out_ptr,
    N, C, H, W, G, CPG, eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, g)
    n = pid // G
    g = pid % G

    HW = H * W
    group_size = CPG * HW
    base = n * C * HW + g * CPG * HW

    # Two-pass: compute mean and var of sigmoid((x + bias) * scale) for this group
    sum_val = 0.0
    sum_sq = 0.0

    for i in range(0, group_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_size
        # channel index within group
        c_in_group = offs // HW
        c_global = g * CPG + c_in_group
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        v = (x + b) * s
        v = tl.sigmoid(v)
        v = tl.where(mask, v, 0.0)
        sum_val += tl.sum(v)
        sum_sq += tl.sum(v * v)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize and apply affine
    for i in range(0, group_size, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < group_size
        c_in_group = offs // HW
        c_global = g * CPG + c_in_group
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(scale_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        gw = tl.load(gn_w_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        gb = tl.load(gn_b_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
        v = (x + b) * s
        v = tl.sigmoid(v)
        v = (v - mean) * rstd * gw + gb
        tl.store(out_ptr + base + offs, v, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        N, C, H, W = x.shape
        G = self.num_groups
        CPG = C // G
        out = torch.empty_like(x)
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps

        group_size = CPG * H * W
        # Choose BLOCK_SIZE
        if group_size >= 8192:
            BLOCK_SIZE = 2048
            num_warps = 8
        elif group_size >= 2048:
            BLOCK_SIZE = 1024
            num_warps = 4
        else:
            BLOCK_SIZE = 512
            num_warps = 4

        grid = (N * G,)
        fused_bias_scale_sigmoid_gn_kernel[grid](
            x, bias_flat, scale_flat, gn_w, gn_b, out,
            N, C, H, W, G, CPG, eps,
            BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps,
        )
        return out