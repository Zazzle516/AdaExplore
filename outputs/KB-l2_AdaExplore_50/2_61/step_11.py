import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, SPATIAL, GROUPS,
    eps,
    CHANS_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program per (batch, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CHANS_PER_GROUP * SPATIAL
    base = pid_n * C * SPATIAL + pid_g * CHANS_PER_GROUP * SPATIAL
    c_base = pid_g * CHANS_PER_GROUP

    c_off = tl.arange(0, CHANS_PER_GROUP)  # [C']

    # pass 1: mean/var via 2D tile (CHANS_PER_GROUP, BLOCK_S)
    sum_x = tl.zeros((CHANS_PER_GROUP,), dtype=tl.float32)
    sum_x2 = tl.zeros((CHANS_PER_GROUP,), dtype=tl.float32)
    for off in range(0, SPATIAL, BLOCK_S):
        s_idx = off + tl.arange(0, BLOCK_S)
        mask_s = s_idx < SPATIAL
        addrs = base + c_off[:, None] * SPATIAL + s_idx[None, :]
        mask = mask_s[None, :]
        v = tl.load(x_ptr + addrs, mask=mask, other=0.0)
        v = tl.maximum(v, 0.0)
        v = tl.where(mask, v, 0.0)
        sum_x += tl.sum(v, axis=1)
        sum_x2 += tl.sum(v * v, axis=1)

    total_sum = tl.sum(sum_x, axis=0)
    total_sum2 = tl.sum(sum_x2, axis=0)
    inv = 1.0 / group_size
    mean = total_sum * inv
    var = total_sum2 * inv - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # load affine per channel
    w = tl.load(weight_ptr + c_base + c_off)
    b = tl.load(bias_ptr + c_base + c_off)
    scale = rstd * w  # [C']
    shift = b - mean * scale  # [C']

    # pass 2: normalize
    for off in range(0, SPATIAL, BLOCK_S):
        s_idx = off + tl.arange(0, BLOCK_S)
        mask_s = s_idx < SPATIAL
        addrs = base + c_off[:, None] * SPATIAL + s_idx[None, :]
        mask = mask_s[None, :]
        v = tl.load(x_ptr + addrs, mask=mask, other=0.0)
        v = tl.maximum(v, 0.0)
        y = v * scale[:, None] + shift[:, None]
        tl.store(out_ptr + addrs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    SPATIAL = D * H * W
    CHANS_PER_GROUP = C // groups
    x = x.contiguous()
    out = torch.empty_like(x)

    BLOCK_S = 1024
    grid = (N, groups)
    fused_relu_groupnorm_kernel[grid](
        x, out, weight, bias,
        N, C, SPATIAL, groups,
        eps,
        CHANS_PER_GROUP=CHANS_PER_GROUP,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=2,
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