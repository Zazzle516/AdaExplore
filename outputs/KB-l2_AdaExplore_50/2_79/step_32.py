import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def mean_invstd_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr,
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    base = pid_n * C * S + pid_c * S
    m = tl.load(mult_ptr + pid_c)

    sum_val = 0.0
    sumsq_val = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum_val += tl.sum(y, axis=0)
        sumsq_val += tl.sum(y * y, axis=0)

    mean = sum_val / S
    var = sumsq_val / S - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * C + pid_c, mean)
    tl.store(invstd_ptr + pid_n * C + pid_c, inv_std)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=16, num_stages=2),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr, out_ptr,
    N, S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    neg_inf = float('-inf')
    acc = tl.full((BLOCK_S,), neg_inf, dtype=tl.float32)

    n_base = pid_n * C * S
    nc_base = pid_n * C

    for c in tl.static_range(0, C):
        m = tl.load(mult_ptr + c)
        mean = tl.load(mean_ptr + nc_base + c)
        inv_std = tl.load(invstd_ptr + nc_base + c)

        x = tl.load(x_ptr + n_base + c * S + s_offs, mask=mask, other=0.0)
        y = x * m
        y = (y - mean) * inv_std
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
        y = y * m
        acc = tl.maximum(acc, y)

    tl.store(out_ptr + pid_n * S + s_offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N, C, S)
        mult_flat = self.multiplier.contiguous().view(-1)

        mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

        grid1 = (N, C)
        mean_invstd_kernel[grid1](
            x_flat, mult_flat, mean, invstd,
            N, C, S,
            1e-5,
            BLOCK_S=1024,
            num_warps=4,
        )

        out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
        out_flat = out.view(N, S)

        grid2 = lambda meta: (N, (S + meta['BLOCK_S'] - 1) // meta['BLOCK_S'])
        fused_norm_clamp_max_kernel[grid2](
            x_flat, mult_flat, mean, invstd, out_flat,
            N, S,
            self.clamp_min, self.clamp_max,
            C=C,
        )

        return out