import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=8, num_stages=2),
    ],
    key=['S', 'CPG'],
)
@triton.jit
def fused_gn_min_clamp_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    S,
    min_value: tl.constexpr, max_value: tl.constexpr, eps: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)
    G = tl.num_programs(1)
    C = G * CPG

    base = pid_n * C * S + pid_g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0

    # First pass
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG):
            ptr = x_ptr + base + c_idx * S + offs
            vals = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(vals, axis=0)
            sum_sq += tl.sum(vals * vals, axis=0)

    inv_count = 1.0 / (CPG * S)
    mean = sum_val * inv_count
    var = sum_sq * inv_count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize + affine + min + clamp
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_idx in tl.static_range(0, CPG):
            channel = pid_g * CPG + c_idx
            w = tl.load(weight_ptr + channel).to(tl.float32)
            b = tl.load(bias_ptr + channel).to(tl.float32)
            ptr = x_ptr + base + c_idx * S + offs
            vals = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
            norm = (vals - mean) * rstd
            y = norm * w + b
            y = tl.minimum(y, min_value)
            y = tl.maximum(y, min_value)
            y = tl.minimum(y, max_value)
            tl.store(out_ptr + base + c_idx * S + offs, y, mask=mask)


def fused_gn_min_clamp(x, weight, bias, groups, min_value, max_value, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    CPG = C // groups
    x_flat = x.contiguous()
    out = torch.empty_like(x_flat)

    grid = (N, groups)
    fused_gn_min_clamp_kernel[grid](
        x_flat, out, weight, bias,
        S,
        float(min_value), float(max_value), float(eps),
        CPG=CPG,
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