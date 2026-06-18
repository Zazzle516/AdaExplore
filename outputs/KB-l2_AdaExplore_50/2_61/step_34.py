import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    SPATIAL, CHANS_PER_GROUP,
    C,
    eps,
    BLOCK: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    base = pid_n * C * SPATIAL + pid_g * CHANS_PER_GROUP * SPATIAL

    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, GROUP_SIZE, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        v = tl.maximum(v, 0.0)
        sum_x += tl.sum(v, axis=0)
        sum_x2 += tl.sum(v * v, axis=0)

    inv = 1.0 / GROUP_SIZE
    mean = sum_x * inv
    var = sum_x2 * inv - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    for off in range(0, GROUP_SIZE, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < GROUP_SIZE
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        v = tl.maximum(v, 0.0)
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

    GROUP_SIZE = CHANS_PER_GROUP * SPATIAL
    BLOCK = 2048
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x, out, weight, bias,
        SPATIAL, CHANS_PER_GROUP,
        C,
        eps,
        BLOCK=BLOCK,
        GROUP_SIZE=GROUP_SIZE,
        num_warps=8,
        num_stages=3,
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