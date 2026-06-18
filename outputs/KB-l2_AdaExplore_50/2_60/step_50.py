import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
    ],
    key=['group_size'],
)
@triton.jit
def fused_swish_gn_stats_kernel(
    x_ptr, y_ptr, mean_ptr, rstd_ptr,
    N, C, G, CPG, S,
    group_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # n * G + g
    n = pid // G
    g = pid % G
    base = n * C * S + g * CPG * S

    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, group_size, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < group_size
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * tl.sigmoid(x)
        tl.store(y_ptr + base + idx, y, mask=mask)
        sum_val += tl.sum(y, axis=0)
        sum_sq += tl.sum(y * y, axis=0)

    mean = sum_val / group_size
    var = sum_sq / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


@triton.jit
def gn_prep_affine_kernel(
    mean_ptr, rstd_ptr, weight_ptr, bias_ptr,
    A_ptr, B_ptr,
    N, C, G, CPG,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)  # n
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    g_idx = offs_c // CPG
    stat_idx = pid * G + g_idx
    mean = tl.load(mean_ptr + stat_idx, mask=mask_c, other=0.0)
    rstd = tl.load(rstd_ptr + stat_idx, mask=mask_c, other=0.0)
    w = tl.load(weight_ptr + offs_c, mask=mask_c, other=0.0)
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    A = w * rstd
    B = b - mean * A
    tl.store(A_ptr + pid * C + offs_c, A, mask=mask_c)
    tl.store(B_ptr + pid * C + offs_c, B, mask=mask_c)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
    ],
    key=['total'],
)
@triton.jit
def gn_apply_hardswish_kernel(
    x_ptr, out_ptr, A_ptr, B_ptr,
    C, S, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)  # block index over all elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # compute n, c from offsets
    c_idx = (offsets // S) % C
    n_idx = offsets // (C * S)
    nc_idx = n_idx * C + c_idx

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    A = tl.load(A_ptr + nc_idx, mask=mask, other=0.0)
    B = tl.load(B_ptr + nc_idx, mask=mask, other=0.0)

    y = x * A + B
    # hardswish: y * relu6(y + 3) / 6
    t = y + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    out = y * t * (1.0 / 6.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_fused_swish_gn_hardswish(x, weight, bias, groups, eps):
    x = x.contiguous()
    N, C = x.shape[0], x.shape[1]
    spatial = 1
    for s in x.shape[2:]:
        spatial *= s
    G = groups
    CPG = C // G
    group_size = CPG * spatial

    y = torch.empty_like(x)
    mean = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((N * G,), device=x.device, dtype=torch.float32)

    grid_stats = (N * G,)
    fused_swish_gn_stats_kernel[grid_stats](
        x, y, mean, rstd,
        N, C, G, CPG, spatial,
        group_size,
        eps,
    )

    # Precompute per (N, C) affine scale and bias
    A = torch.empty((N, C), device=x.device, dtype=x.dtype)
    B = torch.empty((N, C), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    gn_prep_affine_kernel[(N,)](
        mean, rstd, weight, bias,
        A, B,
        N, C, G, CPG,
        BLOCK_C=BLOCK_C,
    )

    out = torch.empty_like(x)
    total = N * C * spatial
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    gn_apply_hardswish_kernel[grid](
        y, out, A, B,
        C, spatial, total,
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
        x = triton_fused_swish_gn_hardswish(x, self.group_norm.weight, self.group_norm.bias, self.groups, self.eps)
        return x