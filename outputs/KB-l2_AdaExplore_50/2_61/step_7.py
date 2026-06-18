import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_relu_reduce_kernel(
    x_ptr, out_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    mean_ptr, rstd_ptr,
    eps: tl.constexpr,
    inv_group_size: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // GROUPS
    g = pid % GROUPS

    base = n * C * S + g * C_PER_GROUP * S

    sum_x = tl.zeros((BLOCK_S,), dtype=tl.float32)
    sum_x2 = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for c in range(0, C_PER_GROUP):
        c_offset = base + c * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_offset + offs, mask=mask, other=0.0)
            v = tl.maximum(v, 0.0)
            tl.store(out_ptr + c_offset + offs, v, mask=mask)
            sum_x += tl.where(mask, v, 0.0)
            sum_x2 += tl.where(mask, v * v, 0.0)

    total_x = tl.sum(sum_x, axis=0)
    total_x2 = tl.sum(sum_x2, axis=0)
    mean = total_x * inv_group_size
    var = total_x2 * inv_group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def normalize_kernel(
    out_ptr, weight_ptr, bias_ptr, mean_ptr, rstd_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    C_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_ng = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)

    n = pid_ng // GROUPS
    g = pid_ng % GROUPS
    c_idx = g * C_PER_GROUP + pid_c

    mean = tl.load(mean_ptr + pid_ng)
    rstd = tl.load(rstd_ptr + pid_ng)
    w = tl.load(weight_ptr + c_idx)
    b = tl.load(bias_ptr + c_idx)

    offset = n * C * S + c_idx * S + pid_s * BLOCK_S
    offs = tl.arange(0, BLOCK_S)
    mask = (pid_s * BLOCK_S + offs) < S
    v = tl.load(out_ptr + offset + offs, mask=mask, other=0.0)
    y = (v - mean) * rstd * w + b
    tl.store(out_ptr + offset + offs, y, mask=mask)


def fused_relu_groupnorm(x, weight, bias, groups, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    C_PER_GROUP = C // groups
    x_flat = x.contiguous().view(N, C, S)
    out = torch.empty_like(x_flat)

    mean = torch.empty((N * groups,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N * groups,), device=x.device, dtype=torch.float32)

    BLOCK_S = 1024
    group_size = C_PER_GROUP * S
    grid = (N * groups,)
    fused_relu_reduce_kernel[grid](
        x_flat, out,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        BLOCK_S=BLOCK_S,
        mean_ptr=mean, rstd_ptr=rstd,
        eps=eps,
        inv_group_size=1.0 / group_size,
        num_warps=8,
        num_stages=2,
    )

    BLOCK_S2 = 1024
    grid2 = (N * groups, C_PER_GROUP, (S + BLOCK_S2 - 1) // BLOCK_S2)
    normalize_kernel[grid2](
        out, weight, bias, mean, rstd,
        N, C, S,
        GROUPS=groups,
        C_PER_GROUP=C_PER_GROUP,
        BLOCK_S=BLOCK_S2,
        num_warps=4,
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