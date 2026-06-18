import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    S, C_PER_G,
    eps,
    BLOCK_S: tl.constexpr,
    C_PER_G_C: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)
    G = tl.num_programs(1)
    C = G * C_PER_G_C

    base = n * C * S + g * C_PER_G_C * S

    sum_x = 0.0
    sum_x2 = 0.0

    # Pass 1: accumulate sum and sum of squares (post-ReLU)
    for c_off in tl.static_range(0, C_PER_G_C):
        c_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            sum_x += tl.sum(x)
            sum_x2 += tl.sum(x * x)

    inv_gs = 1.0 / GROUP_SIZE
    mean = sum_x * inv_gs
    var = sum_x2 * inv_gs - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Pass 2: normalize and write
    for c_off in tl.static_range(0, C_PER_G_C):
        c_idx = g * C_PER_G_C + c_off
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        scale = w * rstd
        shift = b - mean * scale
        c_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            x = tl.maximum(x, 0.0)
            y = x * scale + shift
            tl.store(y_ptr + c_base + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_G = C // groups
    x_c = x.contiguous()
    y = torch.empty_like(x_c)

    BLOCK_S = 2048
    GROUP_SIZE = C_PER_G * S
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x_c, y, weight, bias,
        S, C_PER_G,
        eps,
        BLOCK_S=BLOCK_S,
        C_PER_G_C=C_PER_G,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=8,
        num_stages=3,
    )
    return y


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