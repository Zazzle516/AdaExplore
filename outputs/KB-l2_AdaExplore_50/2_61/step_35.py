import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_groupnorm_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    group_size = C_PER_GROUP * S
    base = n * C * S + g * C_PER_GROUP * S

    sum_x = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK_S], dtype=tl.float32)

    for c in range(0, C_PER_GROUP):
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            sum_x += tl.where(mask, v, 0.0)
            sum_x2 += tl.where(mask, v * v, 0.0)

    s_x = tl.sum(sum_x, axis=0)
    s_x2 = tl.sum(sum_x2, axis=0)

    mean = s_x / group_size
    var = s_x2 / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def fused_relu_groupnorm_apply_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr, mean_ptr, rstd_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # grid: (N*C, ceil(S/BLOCK_S))
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)

    n = pid_nc // C
    c = pid_nc % C
    g = c // C_PER_GROUP

    stats_idx = n * GROUPS + g
    mean = tl.load(mean_ptr + stats_idx)
    rstd = tl.load(rstd_ptr + stats_idx)
    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)

    base = n * C * S + c * S
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = offs < S
    v = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    v = tl.maximum(v, 0.0)
    y = (v - mean) * rstd * w + b
    tl.store(out_ptr + base + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)

    mean = torch.empty((N * groups,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N * groups,), device=x.device, dtype=torch.float32)

    BLOCK_S_STATS = 1024
    grid_stats = (N * groups,)
    fused_relu_groupnorm_stats_kernel[grid_stats](
        x_flat, mean, rstd,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        eps=eps,
        BLOCK_S=BLOCK_S_STATS,
        num_warps=4,
    )

    BLOCK_S_APPLY = 1024
    grid_apply = (N * C, (S + BLOCK_S_APPLY - 1) // BLOCK_S_APPLY)
    fused_relu_groupnorm_apply_kernel[grid_apply](
        x_flat, out, weight, bias, mean, rstd,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        BLOCK_S=BLOCK_S_APPLY,
        num_warps=4,
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