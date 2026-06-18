import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, SPATIAL, GROUPS, CHANS_PER_GROUP,
    eps,
    BLOCK: tl.constexpr,
):
    # one program per (batch, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CHANS_PER_GROUP * SPATIAL
    base = pid_n * C * SPATIAL + pid_g * group_size

    # pass 1: mean and var with ReLU applied
    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        v = tl.maximum(v, 0.0)
        sum_x += tl.sum(tl.where(mask, v, 0.0), axis=0)
        sum_x2 += tl.sum(tl.where(mask, v * v, 0.0), axis=0)

    inv = 1.0 / group_size
    mean = sum_x * inv
    var = sum_x2 * inv - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # pass 2: normalize, apply affine
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        v = tl.maximum(v, 0.0)
        # channel index inside group
        c_in_group = idx // SPATIAL
        c_global = pid_g * CHANS_PER_GROUP + c_in_group
        w = tl.load(weight_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean) * rstd * w + b
        tl.store(out_ptr + base + idx, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    SPATIAL = D * H * W
    CHANS_PER_GROUP = C // groups
    x = x.contiguous()
    out = torch.empty_like(x)

    group_size = CHANS_PER_GROUP * SPATIAL
    BLOCK = 1024
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x, out, weight, bias,
        N, C, SPATIAL, groups, CHANS_PER_GROUP,
        eps,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out


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