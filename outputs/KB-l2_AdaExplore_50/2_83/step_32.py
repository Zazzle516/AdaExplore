import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S, G,
    min_value, max_value,
    eps,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    c_start = g * CPG
    group_size = CPG * S

    # Pointers for the group
    base = n * C * S + c_start * S

    # First pass: compute sum and sum_sq
    sum_val = 0.0
    sum_sq = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask_s = offs < S
        for c_off in tl.static_range(0, CPG):
            ptr = x_ptr + base + c_off * S + offs
            v = tl.load(ptr, mask=mask_s, other=0.0)
            sum_val += tl.sum(v, axis=0)
            sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Load weights/biases for this group's channels
    # Second pass: normalize and write
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask_s = offs < S
        for c_off in tl.static_range(0, CPG):
            c = c_start + c_off
            w = tl.load(weight_ptr + c)
            b = tl.load(bias_ptr + c)
            ptr = x_ptr + base + c_off * S + offs
            v = tl.load(ptr, mask=mask_s, other=0.0)
            norm = (v - mean) * rstd * w + b
            norm = tl.minimum(norm, min_value)
            norm = tl.maximum(norm, min_value)
            norm = tl.minimum(norm, max_value)
            tl.store(out_ptr + base + c_off * S + offs, norm, mask=mask_s)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, min_value, max_value, dropout_p):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.norm = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout_p)
        self.groups = groups
        self.out_channels = out_channels
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N, C, S)
        out = torch.empty_like(x_flat)

        G = self.groups
        CPG = C // G

        grid = (N * G,)
        fused_gn_min_clamp_kernel[grid](
            x_flat, out,
            self.norm.weight, self.norm.bias,
            N, C, S, G,
            self.min_value, self.max_value,
            self.eps,
            CPG=CPG,
        )
        out = out.view(N, C, D, H, W)
        out = self.dropout(out)
        return out