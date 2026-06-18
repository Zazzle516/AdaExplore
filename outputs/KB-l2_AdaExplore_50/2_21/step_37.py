import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['SPATIAL', 'CH_PER_GROUP'],
)
@triton.jit
def fused_bias_scale_sigmoid_groupnorm_kernel(
    x_ptr, out_ptr,
    bias_ptr, scale_ptr,
    gn_weight_ptr, gn_bias_ptr,
    N, C, H, W,
    GROUPS, CH_PER_GROUP: tl.constexpr,
    eps,
    SPATIAL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    base = n * C * SPATIAL + g * CH_PER_GROUP * SPATIAL
    total = GROUP_SIZE  # = CH_PER_GROUP * SPATIAL

    # Hoist bias/scale loads for the group: load CH_PER_GROUP values into a vector.
    c_offs = g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)
    b_vec = tl.load(bias_ptr + c_offs)
    s_vec = tl.load(scale_ptr + c_offs)
    gw_vec = tl.load(gn_weight_ptr + c_offs)
    gb_vec = tl.load(gn_bias_ptr + c_offs)

    # First pass: compute sigmoid((x+bias)*scale), accumulate stats
    sum_val = 0.0
    sum_sq = 0.0

    for ci in tl.static_range(0, CH_PER_GROUP):
        b = tl.sum(tl.where(tl.arange(0, CH_PER_GROUP) == ci, b_vec, 0.0))
        s = tl.sum(tl.where(tl.arange(0, CH_PER_GROUP) == ci, s_vec, 0.0))
        ch_base = base + ci * SPATIAL
        for off in range(0, SPATIAL, BLOCK_SIZE):
            idx = off + tl.arange(0, BLOCK_SIZE)
            mask = idx < SPATIAL
            x = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            y = (x + b) * s
            y = tl.sigmoid(y)
            y = tl.where(mask, y, 0.0)
            sum_val += tl.sum(y)
            sum_sq += tl.sum(y * y)

    mean = sum_val / total
    var = sum_sq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: recompute sigmoid, normalize, apply affine
    for ci in tl.static_range(0, CH_PER_GROUP):
        b = tl.sum(tl.where(tl.arange(0, CH_PER_GROUP) == ci, b_vec, 0.0))
        s = tl.sum(tl.where(tl.arange(0, CH_PER_GROUP) == ci, s_vec, 0.0))
        gw = tl.sum(tl.where(tl.arange(0, CH_PER_GROUP) == ci, gw_vec, 0.0))
        gb = tl.sum(tl.where(tl.arange(0, CH_PER_GROUP) == ci, gb_vec, 0.0))
        ch_base = base + ci * SPATIAL
        for off in range(0, SPATIAL, BLOCK_SIZE):
            idx = off + tl.arange(0, BLOCK_SIZE)
            mask = idx < SPATIAL
            x = tl.load(x_ptr + ch_base + idx, mask=mask, other=0.0)
            y = (x + b) * s
            y = tl.sigmoid(y)
            y = (y - mean) * rstd
            y = y * gw + gb
            tl.store(out_ptr + ch_base + idx, y, mask=mask)


def fused_bsg_gn(x, bias, scale, gn_weight, gn_bias, num_groups, eps=1e-5):
    N, C, H, W = x.shape
    SPATIAL = H * W
    CH_PER_GROUP = C // num_groups
    GROUP_SIZE = CH_PER_GROUP * SPATIAL

    out = torch.empty_like(x)

    grid = (N * num_groups,)
    fused_bias_scale_sigmoid_groupnorm_kernel[grid](
        x, out,
        bias, scale,
        gn_weight, gn_bias,
        N, C, H, W,
        num_groups, CH_PER_GROUP,
        eps,
        SPATIAL=SPATIAL,
        GROUP_SIZE=GROUP_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups, bias_shape, scale_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale = nn.Parameter(torch.randn(scale_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        scale_flat = self.scale.view(-1).contiguous()
        gn_w = self.group_norm.weight.contiguous()
        gn_b = self.group_norm.bias.contiguous()
        eps = self.group_norm.eps
        out = fused_bsg_gn(x, bias_flat, scale_flat, gn_w, gn_b, self.num_groups, eps)
        return out