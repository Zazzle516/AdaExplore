import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_gelu_groupnorm_stats_kernel(
    x_ptr, y_ptr, mean_ptr, rstd_ptr,
    N, C, HW, G, CPG,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Pass 1: compute gelu(x), write to y, accumulate stats per (n, g)
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_elems = CPG * HW
    base = n * C * HW + g * CPG * HW

    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    sumsq_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    inv_sqrt2 = 0.7071067811865475

    for off in range(0, group_elems, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_elems
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        gx = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
        gx_masked = tl.where(mask, gx, 0.0)
        sum_val += gx_masked
        sumsq_val += gx_masked * gx_masked
        tl.store(y_ptr + base + idx, gx, mask=mask)

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sumsq_val, axis=0)

    inv_n = 1.0 / group_elems
    mean = s * inv_n
    var = sq * inv_n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def groupnorm_apply_kernel(
    y_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
    N, C, HW, G, CPG,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per (n, g, tile of group_elems)
    pid_ng = tl.program_id(0)
    pid_t = tl.program_id(1)

    n = pid_ng // G
    g = pid_ng % G

    group_elems = CPG * HW
    base = n * C * HW + g * CPG * HW

    mean = tl.load(mean_ptr + pid_ng)
    rstd = tl.load(rstd_ptr + pid_ng)

    off = pid_t * BLOCK_SIZE
    idx = off + tl.arange(0, BLOCK_SIZE)
    mask = idx < group_elems

    x = tl.load(y_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)

    c_in_group = idx // HW
    c_global = g * CPG + c_in_group
    w = tl.load(weight_ptr + c_global, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + c_global, mask=mask, other=0.0).to(tl.float32)

    y = (x - mean) * rstd * w + b
    tl.store(y_ptr + base + idx, y, mask=mask)


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

        # Pass 1: gelu + stats
        if group_elems >= 8192:
            BLOCK_SIZE_P1 = 2048
            num_warps_p1 = 8
        elif group_elems >= 4096:
            BLOCK_SIZE_P1 = 1024
            num_warps_p1 = 8
        elif group_elems >= 1024:
            BLOCK_SIZE_P1 = 512
            num_warps_p1 = 4
        else:
            BLOCK_SIZE_P1 = 256
            num_warps_p1 = 4

        mean_buf = torch.empty((N * G,), device=x.device, dtype=torch.float32)
        rstd_buf = torch.empty((N * G,), device=x.device, dtype=torch.float32)

        grid1 = (N * G,)
        fused_gelu_groupnorm_stats_kernel[grid1](
            x, y, mean_buf, rstd_buf,
            N, C, HW, G, CPG,
            self.eps,
            BLOCK_SIZE=BLOCK_SIZE_P1,
            num_warps=num_warps_p1,
            num_stages=3,
        )

        # Pass 2: normalize+affine (tiled across group_elems for parallelism)
        BLOCK_SIZE_P2 = 2048
        num_warps_p2 = 8
        n_tiles = (group_elems + BLOCK_SIZE_P2 - 1) // BLOCK_SIZE_P2
        grid2 = (N * G, n_tiles)
        groupnorm_apply_kernel[grid2](
            y, mean_buf, rstd_buf,
            self.group_norm.weight, self.group_norm.bias,
            N, C, HW, G, CPG,
            BLOCK_SIZE=BLOCK_SIZE_P2,
            num_warps=num_warps_p2,
            num_stages=3,
        )

        return y