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
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    c_start = g * CPG
    group_size = CPG * S

    base = n * C * S + c_start * S

    # accumulate sum and sum_sq across channels and spatial blocks
    sum_val = tl.zeros((BLOCK_S,), dtype=tl.float32)
    sum_sq = tl.zeros((BLOCK_S,), dtype=tl.float32)

    s_offs_base = tl.arange(0, BLOCK_S)

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + s_offs_base
        mask = offs < S
        for c_off in tl.static_range(0, CPG):
            ptr = x_ptr + base + c_off * S + offs
            v = tl.load(ptr, mask=mask, other=0.0)
            sum_val += v
            sum_sq += v * v

    total_sum = tl.sum(sum_val, axis=0)
    total_sum_sq = tl.sum(sum_sq, axis=0)

    mean = total_sum / group_size
    var = total_sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # second pass: normalize + min + clamp
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + s_offs_base
        mask = offs < S
        for c_off in tl.static_range(0, CPG):
            c = c_start + c_off
            w = tl.load(weight_ptr + c)
            b = tl.load(bias_ptr + c)
            ptr = x_ptr + base + c_off * S + offs
            v = tl.load(ptr, mask=mask, other=0.0)
            norm = (v - mean) * rstd * w + b
            norm = tl.minimum(norm, min_value)
            norm = tl.maximum(norm, min_value)
            norm = tl.minimum(norm, max_value)
            tl.store(out_ptr + base + c_off * S + offs, norm, mask=mask)


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

        BLOCK_S = 1024

        grid = (N * G,)
        fused_gn_min_clamp_kernel[grid](
            x_flat, out,
            self.norm.weight, self.norm.bias,
            N, C, S, G,
            self.min_value, self.max_value,
            self.eps,
            CPG=CPG,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )
        out = out.view(N, C, D, H, W)
        out = self.dropout(out)
        return out