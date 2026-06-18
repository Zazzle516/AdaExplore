import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def swish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def gn_stats_kernel(
    x_ptr, mean_ptr, rstd_ptr,
    N, C, G, CPG, S,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # n * G + g
    n = pid // G
    g = pid % G
    group_size = CPG * S
    base = n * C * S + g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def gn_apply_hardswish_kernel(
    x_ptr, out_ptr, mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
    N, C, G, CPG, S,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # block index over all elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * S
    mask = offsets < total

    # compute n, c, s from offsets
    s_idx = offsets % S
    c_idx = (offsets // S) % C
    n_idx = offsets // (C * S)
    g_idx = c_idx // CPG
    stat_idx = n_idx * G + g_idx

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + stat_idx, mask=mask, other=0.0)
    rstd = tl.load(rstd_ptr + stat_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    normed = (x - mean) * rstd
    y = normed * w + b
    # hardswish: y * relu6(y + 3) / 6
    t = y + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    out = y * t * (1.0 / 6.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_swish(x):
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    swish_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK)
    return out


def triton_groupnorm_hardswish(x, weight, bias, groups, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    spatial = 1
    for s in x.shape[2:]:
        spatial *= s
    G = groups
    CPG = C // G

    mean = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N * G,), device=x.device, dtype=torch.float32)

    BLOCK_STATS = 1024
    grid_stats = (N * G,)
    gn_stats_kernel[grid_stats](
        x, mean, rstd,
        N, C, G, CPG, spatial,
        eps,
        BLOCK_SIZE=BLOCK_STATS,
    )

    out = torch.empty_like(x)
    total = N * C * spatial
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    gn_apply_hardswish_kernel[grid](
        x, out, mean, rstd, weight, bias,
        N, C, G, CPG, spatial,
        BLOCK_SIZE=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, eps, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=out_channels, eps=eps)
        self.groups = groups
        self.eps = eps

    def forward(self, x):
        x = self.conv_transpose(x)
        x = triton_swish(x)
        x = triton_groupnorm_hardswish(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x