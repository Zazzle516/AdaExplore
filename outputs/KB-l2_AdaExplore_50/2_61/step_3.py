import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Each program handles one (batch, group)
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C_PER_GROUP * S  # number of elements in this group
    base = n * C * S + g * C_PER_GROUP * S

    # Compute mean and variance over (C_PER_GROUP, S) after ReLU
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

    mean = sum_x / group_size
    var = sum_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize
    for c in range(0, C_PER_GROUP):
        c_idx = g * C_PER_GROUP + c
        w = tl.load(weight_ptr + c_idx)
        b = tl.load(bias_ptr + c_idx)
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            y = (v - mean) * rstd * w + b
            tl.store(out_ptr + c_offset + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)

    BLOCK_S = 1024
    grid = (N * groups,)
    fused_relu_groupnorm_kernel[grid](
        x_flat, out, weight, bias,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        eps=eps,
        BLOCK_S=BLOCK_S,
        num_warps=4,
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