import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S, G, CPG,
    min_value, max_value, eps,
    BLOCK_S: tl.constexpr,
    CPG_C: tl.constexpr,
):
    # one program per (batch, group)
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    base = pid_n * C * S + pid_g * CPG * S

    # First pass: compute sum and sum of squares (single pass per element)
    sum_val = 0.0
    sum_sq = 0.0

    for c_idx in tl.static_range(0, CPG_C):
        c_offset = base + c_idx * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(vals, axis=0)
            sum_sq += tl.sum(vals * vals, axis=0)

    inv_count = 1.0 / (CPG * S)
    mean = sum_val * inv_count
    var = sum_sq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize + affine + min + clamp
    for c_idx in tl.static_range(0, CPG_C):
        channel = pid_g * CPG + c_idx
        w = tl.load(weight_ptr + channel).to(tl.float32)
        b = tl.load(bias_ptr + channel).to(tl.float32)
        c_offset = base + c_idx * S
        wm = w * rstd
        bm = b - mean * wm
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            vals = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0).to(tl.float32)
            y = vals * wm + bm
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

    # pick BLOCK_S
    BLOCK_S = 4096
    # CPG_C must be a constexpr >= CPG
    CPG_C = CPG  # since CPG = 16/8 = 2

    grid = (N, groups)
    fused_gn_min_clamp_kernel[grid](
        x_flat, out, weight, bias,
        N, C, S, groups, CPG,
        float(min_value), float(max_value), float(eps),
        BLOCK_S=BLOCK_S,
        CPG_C=CPG_C,
        num_warps=4,
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