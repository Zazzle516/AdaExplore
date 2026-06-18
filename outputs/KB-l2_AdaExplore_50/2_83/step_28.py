import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    N, C, S, G, CPG,
    min_value, max_value,
    eps,
    BLOCK_S: tl.constexpr,
    CPG_C: tl.constexpr,
):
    # one program per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    # group start channel
    c_start = g * CPG
    group_size = CPG * S

    # compute mean and var
    sum_val = 0.0
    sum_sq = 0.0
    for c_off in range(0, CPG_C):
        c = c_start + c_off
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = (offs < S) & (c_off < CPG)
            ptr = x_ptr + n * C * S + c * S + offs
            v = tl.load(ptr, mask=mask, other=0.0)
            sum_val += tl.sum(v, axis=0)
            sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # normalize, apply affine, then min and clamp
    for c_off in range(0, CPG_C):
        c = c_start + c_off
        if c_off < CPG:
            w = tl.load(weight_ptr + c)
            b = tl.load(bias_ptr + c)
            for s_start in range(0, S, BLOCK_S):
                offs = s_start + tl.arange(0, BLOCK_S)
                mask = offs < S
                ptr = x_ptr + n * C * S + c * S + offs
                v = tl.load(ptr, mask=mask, other=0.0)
                norm = (v - mean) * rstd * w + b
                # min with min_value
                norm = tl.minimum(norm, min_value)
                # clamp
                norm = tl.maximum(norm, min_value)
                norm = tl.minimum(norm, max_value)
                tl.store(out_ptr + n * C * S + c * S + offs, norm, mask=mask)


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
        # next power of two >= CPG for constexpr unroll
        CPG_C = 1
        while CPG_C < CPG:
            CPG_C *= 2

        BLOCK_S = 1024

        grid = (N * G,)
        fused_gn_min_clamp_kernel[grid](
            x_flat, out,
            self.norm.weight, self.norm.bias,
            N, C, S, G, CPG,
            self.min_value, self.max_value,
            self.eps,
            BLOCK_S=BLOCK_S,
            CPG_C=CPG_C,
            num_warps=4,
        )
        out = out.view(N, C, D, H, W)
        out = self.dropout(out)
        return out