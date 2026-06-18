import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gelu_groupnorm_kernel(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    HW, CPG: tl.constexpr, group_elems,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    G = tl.num_programs(1)
    C = G * CPG

    base = pid * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865475

    # First pass: compute sum and sumsq of GELU(x)
    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gx_masked = tl.where(mask, gx, 0.0)
        sum_val += gx_masked
        sumsq_val += gx_masked * gx_masked

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    inv_n = 1.0 / group_elems
    mean = s * inv_n
    var = sq * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Preload affine params for this group's CPG channels
    c_offs = g * CPG + tl.arange(0, CPG)
    w_vec = tl.load(weight_ptr + c_offs)
    b_vec = tl.load(bias_ptr + c_offs)
    # Combine into scale/shift
    scale_vec = rstd * w_vec
    shift_vec = b_vec - mean * scale_vec  # so y = gx * scale + shift

    # Second pass: write GELU + groupnorm
    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))

        c_in_group = idx // HW
        scale = tl.load(scale_vec_addr_helper(scale_vec, c_in_group), mask=mask) if False else tl.gather(scale_vec, c_in_group)  # placeholder

        # Use tl.gather emulation through indexing-by-load is not available; instead use a loop-free approach:
        # Convert to per-element using a manual gather via where-trees only when CPG is small.
        # Fallback: compute scale/shift per element using selection.
        y = gx * scale  # unused
        tl.store(y_ptr + base + idx, y, mask=mask)


@triton.jit
def fused_gelu_groupnorm_kernel_v2(
    x_ptr, y_ptr, weight_ptr, bias_ptr,
    HW, group_elems,
    eps,
    CPG: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    g = tl.program_id(1)
    G = tl.num_programs(1)
    C = G * CPG

    base = pid * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865475

    # First pass: compute sum and sumsq of GELU(x)
    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gx_masked = tl.where(mask, gx, 0.0)
        sum_val += gx_masked
        sumsq_val += gx_masked * gx_masked

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    inv_n = 1.0 / group_elems
    mean = s * inv_n
    var = sq * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    # Per-channel inner loop in pass 2 to avoid integer-divide gather
    g_off = g * CPG
    for c in tl.static_range(0, CPG):
        c_base = base + c * HW
        w = tl.load(weight_ptr + g_off + c)
        b = tl.load(bias_ptr + g_off + c)
        scale = rstd * w
        shift = b - mean * scale
        for off in range(0, HW, BLOCK_SIZE):
            offs = off + tl.arange(0, BLOCK_SIZE)
            mask = offs < HW
            x = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
            y = gx * scale + shift
            tl.store(y_ptr + c_base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, num_groups):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, H, W = x.shape
        HW = H * W
        G = self.num_groups
        CPG = C // G

        x = x.contiguous()
        y = torch.empty_like(x)

        group_elems = CPG * HW

        # Choose BLOCK_SIZE: prefer larger blocks for big HW
        if HW >= 16384:
            BLOCK_SIZE = 2048
            num_warps = 8
            num_stages = 2
        elif HW >= 4096:
            BLOCK_SIZE = 1024
            num_warps = 8
            num_stages = 2
        elif HW >= 1024:
            BLOCK_SIZE = 512
            num_warps = 4
            num_stages = 2
        else:
            BLOCK_SIZE = 256
            num_warps = 4
            num_stages = 2

        grid = (N, G)
        fused_gelu_groupnorm_kernel_v2[grid](
            x, y,
            self.group_norm.weight, self.group_norm.bias,
            HW, group_elems,
            self.eps,
            CPG=CPG,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return y