import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _relu_partial_stats_kernel(
    x_ptr, tmp_ptr, partial_sum_ptr, partial_sq_ptr,
    GROUP_NUMEL,
    C_PER_GROUP: tl.constexpr,
    S: tl.constexpr,
    BLOCK_S: tl.constexpr,
    NBLOCKS: tl.constexpr,
):
    # 2D grid: (N*GROUPS, NBLOCKS)
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    base = pid_g * C_PER_GROUP * S

    sum_x = tl.zeros((BLOCK_S,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_S,), dtype=tl.float32)

    # Each program handles a chunk of channels: split C_PER_GROUP into NBLOCKS pieces
    C_PER_BLOCK: tl.constexpr = C_PER_GROUP // NBLOCKS
    c_start = pid_b * C_PER_BLOCK

    for c_off in range(0, C_PER_BLOCK):
        c = c_start + c_off
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            tl.store(tmp_ptr + c_offset + offs, v, mask=mask)
            sum_x += tl.where(mask, v, 0.0)
            sum_x2 += tl.where(mask, v * v, 0.0)

    s = tl.sum(sum_x, axis=0)
    s2 = tl.sum(sum_x2, axis=0)
    tl.store(partial_sum_ptr + pid_g * NBLOCKS + pid_b, s)
    tl.store(partial_sq_ptr + pid_g * NBLOCKS + pid_b, s2)


@triton.jit
def _normalize_kernel(
    tmp_ptr, out_ptr, weight_ptr, bias_ptr,
    partial_sum_ptr, partial_sq_ptr,
    GROUPS,
    C_PER_GROUP: tl.constexpr,
    S: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NBLOCKS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_g = tl.program_id(0)
    pid_b = tl.program_id(1)
    g = pid_g % GROUPS
    base = pid_g * C_PER_GROUP * S

    # Load partial sums and reduce
    poffs = tl.arange(0, NBLOCKS)
    sums = tl.load(partial_sum_ptr + pid_g * NBLOCKS + poffs)
    sqs = tl.load(partial_sq_ptr + pid_g * NBLOCKS + poffs)
    total_sum = tl.sum(sums, axis=0)
    total_sq = tl.sum(sqs, axis=0)

    inv_n = 1.0 / GROUP_SIZE
    mean = total_sum * inv_n
    var = total_sq * inv_n - mean * mean
    rstd = tl.rsqrt(var + eps)

    C_PER_BLOCK: tl.constexpr = C_PER_GROUP // NBLOCKS
    c_start = pid_b * C_PER_BLOCK
    weight_base = g * C_PER_GROUP

    for c_off in range(0, C_PER_BLOCK):
        c = c_start + c_off
        w = tl.load(weight_ptr + weight_base + c)
        b = tl.load(bias_ptr + weight_base + c)
        scale = w * rstd
        shift = b - mean * scale
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(tmp_ptr + c_offset + offs, mask=mask, other=0.0)
            y = v * scale + shift
            tl.store(out_ptr + c_offset + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    GROUP_SIZE = C_PER_GROUP * S
    x_flat = x.contiguous().view(N * groups, C_PER_GROUP, S)
    out = torch.empty_like(x_flat)
    tmp = torch.empty_like(x_flat)

    # Choose NBLOCKS to parallelize channels within a group
    if C_PER_GROUP % 4 == 0:
        NBLOCKS = 4
    elif C_PER_GROUP % 2 == 0:
        NBLOCKS = 2
    else:
        NBLOCKS = 1

    partial_sum = torch.empty((N * groups, NBLOCKS), device=x.device, dtype=torch.float32)
    partial_sq = torch.empty((N * groups, NBLOCKS), device=x.device, dtype=torch.float32)

    BLOCK_S = 2048
    grid1 = (N * groups, NBLOCKS)
    _relu_partial_stats_kernel[grid1](
        x_flat, tmp, partial_sum, partial_sq,
        GROUP_SIZE,
        C_PER_GROUP=C_PER_GROUP,
        S=S,
        BLOCK_S=BLOCK_S,
        NBLOCKS=NBLOCKS,
        num_warps=8,
        num_stages=2,
    )

    grid2 = (N * groups, NBLOCKS)
    _normalize_kernel[grid2](
        tmp, out, weight, bias,
        partial_sum, partial_sq,
        groups,
        C_PER_GROUP=C_PER_GROUP,
        S=S,
        GROUP_SIZE=GROUP_SIZE,
        NBLOCKS=NBLOCKS,
        BLOCK_S=BLOCK_S,
        eps=eps,
        num_warps=8,
        num_stages=2,
    )
    return out.view(N, C, D, H, W)


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