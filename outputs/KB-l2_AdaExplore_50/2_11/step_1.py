import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_bn_tanh_pool_gn_kernel(
    x_ptr,           # input after conv_transpose: [N, C, H, W]
    out_ptr,         # output: [N, C, H/2, W/2]
    scale_ptr,       # bn fused scale [C]
    shift_ptr,       # bn fused shift [C]
    gn_weight_ptr,   # [C]
    gn_bias_ptr,     # [C]
    N, C, H, W,
    H_out, W_out,
    G,               # num_groups
    C_per_G,         # channels per group
    SPATIAL_PER_G,   # C_per_G * H_out * W_out
    eps,
    BLOCK: tl.constexpr,
    CPG: tl.constexpr,
    HW_OUT: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    # Each group has CPG channels and HW_OUT spatial positions per channel
    # Total elements in group = CPG * HW_OUT
    total = CPG * HW_OUT

    # Compute mean and var via two passes over the group
    sum_val = 0.0
    sum_sq = 0.0

    # We need to compute bn+tanh+maxpool first, store to output, then normalize
    # Strategy: do pool + tanh + bn, write to out, accumulate stats, then second pass to normalize

    # Channel base in original tensor
    c_base = g * CPG

    # Loop over channels in group
    for ci in tl.static_range(0, CPG):
        c = c_base + ci
        scale = tl.load(scale_ptr + c)
        shift = tl.load(shift_ptr + c)

        # Loop over output spatial positions in blocks
        for blk_start in range(0, HW_OUT, BLOCK):
            offs = blk_start + tl.arange(0, BLOCK)
            mask = offs < HW_OUT

            oh = offs // W_out
            ow = offs % W_out
            ih = oh * 2
            iw = ow * 2

            # 4 input positions: (ih, iw), (ih, iw+1), (ih+1, iw), (ih+1, iw+1)
            base = n * C * H * W + c * H * W
            p00 = tl.load(x_ptr + base + ih * W + iw, mask=mask, other=-1e30)
            p01 = tl.load(x_ptr + base + ih * W + (iw + 1), mask=mask & (iw + 1 < W), other=-1e30)
            p10 = tl.load(x_ptr + base + (ih + 1) * W + iw, mask=mask & (ih + 1 < H), other=-1e30)
            p11 = tl.load(x_ptr + base + (ih + 1) * W + (iw + 1), mask=mask & (ih + 1 < H) & (iw + 1 < W), other=-1e30)

            # Apply bn + tanh, then maxpool
            v00 = tl.where(mask, p00 * scale + shift, -1e30)
            v01 = tl.where(mask & (iw + 1 < W), p01 * scale + shift, -1e30)
            v10 = tl.where(mask & (ih + 1 < H), p10 * scale + shift, -1e30)
            v11 = tl.where(mask & (ih + 1 < H) & (iw + 1 < W), p11 * scale + shift, -1e30)

            t00 = (tl.exp(2.0 * v00) - 1.0) / (tl.exp(2.0 * v00) + 1.0)
            t01 = (tl.exp(2.0 * v01) - 1.0) / (tl.exp(2.0 * v01) + 1.0)
            t10 = (tl.exp(2.0 * v10) - 1.0) / (tl.exp(2.0 * v10) + 1.0)
            t11 = (tl.exp(2.0 * v11) - 1.0) / (tl.exp(2.0 * v11) + 1.0)

            m0 = tl.maximum(t00, t01)
            m1 = tl.maximum(t10, t11)
            pooled = tl.maximum(m0, m1)

            # Store to output
            out_base = n * C * H_out * W_out + c * H_out * W_out
            tl.store(out_ptr + out_base + offs, pooled, mask=mask)

            # Accumulate stats
            pooled_f = tl.where(mask, pooled, 0.0)
            sum_val += tl.sum(pooled_f)
            sum_sq += tl.sum(pooled_f * pooled_f)

    mean = sum_val / total
    var = sum_sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize
    for ci in tl.static_range(0, CPG):
        c = c_base + ci
        gw = tl.load(gn_weight_ptr + c)
        gb = tl.load(gn_bias_ptr + c)

        for blk_start in range(0, HW_OUT, BLOCK):
            offs = blk_start + tl.arange(0, BLOCK)
            mask = offs < HW_OUT
            out_base = n * C * H_out * W_out + c * H_out * W_out
            v = tl.load(out_ptr + out_base + offs, mask=mask, other=0.0)
            v = (v - mean) * rstd * gw + gb
            tl.store(out_ptr + out_base + offs, v, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.tanh = nn.Tanh()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        # Conv transpose using torch (highly optimized)
        x = self.conv_transpose(x)

        # Fold BN
        bn = self.batch_norm
        if bn.training:
            # fall back
            x = bn(x)
            x = torch.tanh(x)
            x = self.max_pool(x)
            x = self.group_norm(x)
            return x

        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale

        N, C, H, W = x.shape
        H_out = H // 2
        W_out = W // 2
        G = self.num_groups
        CPG = C // G
        HW_OUT = H_out * W_out

        x = x.contiguous()
        out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

        # Choose BLOCK based on HW_OUT
        BLOCK = 256
        if HW_OUT <= 64:
            BLOCK = 64
        elif HW_OUT <= 256:
            BLOCK = 256
        else:
            BLOCK = 1024

        grid = (N * G,)
        fused_bn_tanh_pool_gn_kernel[grid](
            x, out,
            scale.contiguous(), shift.contiguous(),
            self.group_norm.weight.contiguous(), self.group_norm.bias.contiguous(),
            N, C, H, W,
            H_out, W_out,
            G, CPG, CPG * HW_OUT,
            self.group_norm.eps,
            BLOCK=BLOCK,
            CPG=CPG,
            HW_OUT=HW_OUT,
            num_warps=4,
        )
        return out