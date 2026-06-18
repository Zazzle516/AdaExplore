import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _softmax_tile_stats_kernel(
    x_ptr,
    max_ptr,
    sum_ptr,
    S,
    NTILES,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    TILE: tl.constexpr,
):
    bc = tl.program_id(0)
    t = tl.program_id(1)

    row_start = bc * S
    off_start = t * TILE
    idx = off_start + tl.arange(0, TILE)
    mask = idx < S

    v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
    v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
    v = tl.where(mask, v, -float('inf'))

    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)

    out_off = bc * NTILES + t
    tl.store(max_ptr + out_off, m)
    tl.store(sum_ptr + out_off, s)


@triton.jit
def _softmax_reduce_kernel(
    max_ptr,
    sum_ptr,
    gmax_ptr,
    ginv_ptr,
    NTILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bc = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < NTILES

    m = tl.load(max_ptr + bc * NTILES + idx, mask=mask, other=-float('inf'))
    s = tl.load(sum_ptr + bc * NTILES + idx, mask=mask, other=0.0)

    gmax = tl.max(m, axis=0)
    s_scaled = s * tl.exp(m - gmax)
    s_scaled = tl.where(mask, s_scaled, 0.0)
    total = tl.sum(s_scaled, axis=0)

    tl.store(gmax_ptr + bc, gmax)
    tl.store(ginv_ptr + bc, 1.0 / total)


@triton.jit
def _softmax_finalize_kernel(
    x_ptr,
    gmax_ptr,
    ginv_ptr,
    scale_ptr,
    out_ptr,
    S,
    C,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    TILE: tl.constexpr,
):
    bc = tl.program_id(0)
    t = tl.program_id(1)

    c = bc % C
    row_start = bc * S
    off_start = t * TILE
    idx = off_start + tl.arange(0, TILE)
    mask = idx < S

    gmax = tl.load(gmax_ptr + bc)
    ginv = tl.load(ginv_ptr + bc)
    sv = tl.load(scale_ptr + c)
    factor = ginv * sv

    v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
    v = tl.minimum(tl.maximum(v, CLAMP_MIN), CLAMP_MAX)
    e = tl.exp(v - gmax) * factor
    tl.store(out_ptr + row_start + idx, e, mask=mask)


def fused_clamp_softmax_scale(x, scale, clamp_min, clamp_max):
    B, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    scale_flat = scale.contiguous().view(-1)

    BC = B * C

    TILE = 8192
    NTILES = (S + TILE - 1) // TILE

    max_buf = torch.empty((BC, NTILES), device=x.device, dtype=torch.float32)
    sum_buf = torch.empty((BC, NTILES), device=x.device, dtype=torch.float32)
    gmax = torch.empty((BC,), device=x.device, dtype=torch.float32)
    ginv = torch.empty((BC,), device=x.device, dtype=torch.float32)

    grid1 = (BC, NTILES)
    _softmax_tile_stats_kernel[grid1](
        x_c, max_buf, sum_buf,
        S, NTILES,
        CLAMP_MIN=float(clamp_min), CLAMP_MAX=float(clamp_max),
        TILE=TILE,
        num_warps=8,
        num_stages=2,
    )

    BLOCK_RED = triton.next_power_of_2(NTILES)
    if BLOCK_RED < 16:
        BLOCK_RED = 16
    grid2 = (BC,)
    _softmax_reduce_kernel[grid2](
        max_buf, sum_buf, gmax, ginv,
        NTILES=NTILES,
        BLOCK=BLOCK_RED,
        num_warps=2,
    )

    grid3 = (BC, NTILES)
    _softmax_finalize_kernel[grid3](
        x_c, gmax, ginv, scale_flat, out,
        S, C,
        CLAMP_MIN=float(clamp_min), CLAMP_MAX=float(clamp_max),
        TILE=TILE,
        num_warps=8,
        num_stages=2,
    )

    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        x = fused_clamp_softmax_scale(x, self.scale, self.clamp_min, self.clamp_max)
        return x