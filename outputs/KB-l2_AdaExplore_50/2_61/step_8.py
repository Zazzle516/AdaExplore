import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S,
    C_PER_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    # x is laid out as (N*GROUPS, C_PER_GROUP, S)
    base = pid * C_PER_GROUP * S

    sum_x = 0.0
    sum_x2 = 0.0

    for c in range(0, C_PER_GROUP):
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            sum_x += tl.sum(tl.where(mask, v, 0.0), axis=0)
            sum_x2 += tl.sum(tl.where(mask, v * v, 0.0), axis=0)

    inv_n = 1.0 / GROUP_SIZE
    mean = sum_x * inv_n
    var = sum_x2 * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    # group index for weight/bias
    g = pid % (tl.num_programs(0) // (tl.num_programs(0) // 1))  # placeholder, recompute below
    # Actually we need group index = pid % GROUPS. Pass via runtime: derive from pid mod groups
    # We'll instead pass group via separate arg below.


@triton.jit
def fused_relu_groupnorm_kernel_v2(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S, GROUPS,
    C_PER_GROUP: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    g = pid % GROUPS
    base = pid * C_PER_GROUP * S

    sum_x = 0.0
    sum_x2 = 0.0

    for c in range(0, C_PER_GROUP):
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            sum_x += tl.sum(tl.where(mask, v, 0.0), axis=0)
            sum_x2 += tl.sum(tl.where(mask, v * v, 0.0), axis=0)

    inv_n = 1.0 / GROUP_SIZE
    mean = sum_x * inv_n
    var = sum_x2 * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    weight_base = g * C_PER_GROUP
    for c in range(0, C_PER_GROUP):
        c_idx = weight_base + c
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        scale = w * rstd
        shift = b - mean * scale
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            y = v * scale + shift
            tl.store(out_ptr + c_offset + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    GROUP_SIZE = C_PER_GROUP * S
    x_flat = x.contiguous().view(N * groups, C_PER_GROUP, S)
    out = torch.empty_like(x_flat)

    BLOCK_S = 1024
    grid = (N * groups,)
    fused_relu_groupnorm_kernel_v2[grid](
        x_flat, out, weight, bias,
        S, groups,
        C_PER_GROUP=C_PER_GROUP,
        GROUP_SIZE=GROUP_SIZE,
        eps=eps,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=3,
    )
    return out.view(N, C, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.groups = groups
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_relu_groupnorm(
            x,
            self.group_norm.weight,
            self.group_norm.bias,
            self.groups,
            self.eps,
        )
        return x