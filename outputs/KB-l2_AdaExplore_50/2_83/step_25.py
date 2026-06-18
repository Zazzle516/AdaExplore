import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S, CPG,
    min_value, max_value, eps,
    stride_n, stride_g,
    BLOCK_S: tl.constexpr,
    CPG_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    base = pid_n * stride_n + pid_g * stride_g

    inv_count = 1.0 / (CPG * S)

    sum_val = 0.0
    sum_sq = 0.0

    # First pass: compute sum and sum of squares
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG_C):
            c_offset = base + c_idx * S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            sum_val += tl.sum(vals, axis=0)
            sum_sq += tl.sum(vals * vals, axis=0)

    mean = sum_val * inv_count
    var = sum_sq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pre-load weights/biases for this group
    # Second pass: normalize + affine + min + clamp
    for c_idx in tl.static_range(0, CPG_C):
        channel = pid_g * CPG_C + c_idx
        w = tl.load(weight_ptr + channel)
        b = tl.load(bias_ptr + channel)
        # Combined scale and shift: y = (x - mean) * rstd * w + b
        # = x * (rstd * w) + (b - mean * rstd * w)
        scale = rstd * w
        shift = b - mean * scale
        c_offset = base + c_idx * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            y = vals * scale + shift
            y = tl.minimum(y, min_value)
            y = tl.maximum(y, min_value)
            y = tl.minimum(y, max_value)
            tl.store(out_ptr + c_offset + offs, y, mask=mask)


def fused_gn_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // groups
    x_flat = x.contiguous()
    out = torch.empty_like(x_flat)

    BLOCK_S = 2048
    CPG_C = CPG

    stride_n = C * S
    stride_g = CPG * S

    grid = (N, groups)
    fused_gn_min_clamp_kernel[grid](
        x_flat, out, weight, bias,
        S, CPG,
        float(min_value), float(max_value), float(eps),
        stride_n, stride_g,
        BLOCK_S=BLOCK_S,
        CPG_C=CPG_C,
        num_warps=8,
        num_stages=2,
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
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        x = fused_gn_min_clamp(
            x, self.norm.weight, self.norm.bias,
            self.groups, self.min_value, self.max_value, self.eps
        )
        x = self.dropout(x)
        return x