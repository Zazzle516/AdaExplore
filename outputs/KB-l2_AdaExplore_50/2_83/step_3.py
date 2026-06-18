import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S,  # S = D*H*W
    groups, channels_per_group,
    min_value, max_value, eps,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, group)
    pid = tl.program_id(0)
    n = pid // groups
    g = pid % groups

    group_size = channels_per_group * S
    base = n * C * S + g * channels_per_group * S

    # First pass: compute mean and var
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize + affine + min + clamp
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        # channel index within the tensor
        c_in_group = idx // S
        c = g * channels_per_group + c_in_group
        w = tl.load(weight_ptr + c, mask=mask, other=0.0)
        b = tl.load(bias_ptr + c, mask=mask, other=0.0)
        y = (x - mean) * rstd * w + b
        # min(y, min_value)
        y = tl.minimum(y, min_value)
        # clamp(y, min_value, max_value)
        y = tl.maximum(y, min_value)
        y = tl.minimum(y, max_value)
        tl.store(out_ptr + base + idx, y, mask=mask)


def fused_gn_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    channels_per_group = C // groups
    x = x.contiguous()
    out = torch.empty_like(x)

    group_size = channels_per_group * S
    # choose block size
    BLOCK_SIZE = 1024
    while BLOCK_SIZE > group_size and BLOCK_SIZE > 64:
        BLOCK_SIZE //= 2
    grid = (N * groups,)
    fused_gn_min_clamp_kernel[grid](
        x, out, weight, bias,
        N, C, S, groups, channels_per_group,
        float(min_value), float(max_value), float(eps),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.groups = groups
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        x = self.conv(x)
        x = fused_gn_min_clamp(
            x,
            self.norm.weight,
            self.norm.bias,
            self.groups,
            self.min_value,
            self.max_value,
            self.norm.eps,
        )
        x = self.dropout(x)
        return x