import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def softmax_partial_kernel(
    x_ptr, partial_max_ptr, partial_sum_ptr,
    S, NBLOCKS,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    blk = tl.program_id(2)
    C = tl.num_programs(1)

    row_start = (b * C + c) * S
    off = blk * BLOCK
    idx = off + tl.arange(0, BLOCK)
    mask = idx < S
    x = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
    x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)

    out_off = (b * C + c) * NBLOCKS + blk
    tl.store(partial_max_ptr + out_off, m)
    tl.store(partial_sum_ptr + out_off, s)


@triton.jit
def softmax_finalize_kernel(
    partial_max_ptr, partial_sum_ptr,
    global_max_ptr, global_inv_sum_ptr,
    scale_ptr,
    NBLOCKS: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    C = tl.num_programs(1)

    off = (b * C + c) * NBLOCKS + tl.arange(0, NBLOCKS)
    m = tl.load(partial_max_ptr + off)
    s = tl.load(partial_sum_ptr + off)
    gmax = tl.max(m, axis=0)
    s_rescaled = s * tl.exp(m - gmax)
    gsum = tl.sum(s_rescaled, axis=0)
    scale = tl.load(scale_ptr + c)
    inv = (1.0 / gsum) * scale

    tl.store(global_max_ptr + b * C + c, gmax)
    tl.store(global_inv_sum_ptr + b * C + c, inv)


@triton.jit
def softmax_write_kernel(
    x_ptr, global_max_ptr, global_inv_sum_ptr, out_ptr,
    S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    blk = tl.program_id(2)
    C = tl.num_programs(1)

    row_start = (b * C + c) * S
    gmax = tl.load(global_max_ptr + b * C + c)
    coef = tl.load(global_inv_sum_ptr + b * C + c)

    off = blk * BLOCK
    idx = off + tl.arange(0, BLOCK)
    mask = idx < S
    x = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
    x = tl.minimum(tl.maximum(x, clamp_min), clamp_max)
    y = tl.exp(x - gmax) * coef
    tl.store(out_ptr + row_start + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv_transpose(x)
        b, c, d, h, w = x.shape
        S = d * h * w

        x_flat = x.contiguous().view(b, c, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(c).contiguous()

        BLOCK = 8192
        NBLOCKS = (S + BLOCK - 1) // BLOCK

        partial_max = torch.empty((b, c, NBLOCKS), device=x.device, dtype=torch.float32)
        partial_sum = torch.empty((b, c, NBLOCKS), device=x.device, dtype=torch.float32)
        global_max = torch.empty((b, c), device=x.device, dtype=torch.float32)
        global_inv = torch.empty((b, c), device=x.device, dtype=torch.float32)

        grid1 = (b, c, NBLOCKS)
        softmax_partial_kernel[grid1](
            x_flat, partial_max, partial_sum,
            S, NBLOCKS,
            clamp_min=self.clamp_min, clamp_max=self.clamp_max,
            BLOCK=BLOCK,
            num_warps=16, num_stages=2,
        )

        # NBLOCKS must be power-of-2 friendly for tl.arange; pad if needed
        # Find next power of 2 >= NBLOCKS
        NB_PADDED = 1
        while NB_PADDED < NBLOCKS:
            NB_PADDED *= 2

        if NB_PADDED != NBLOCKS:
            # pad with -inf for max and 0 for sum
            pad_max = torch.full((b, c, NB_PADDED - NBLOCKS), float('-inf'), device=x.device, dtype=torch.float32)
            pad_sum = torch.zeros((b, c, NB_PADDED - NBLOCKS), device=x.device, dtype=torch.float32)
            partial_max_p = torch.cat([partial_max, pad_max], dim=2).contiguous()
            partial_sum_p = torch.cat([partial_sum, pad_sum], dim=2).contiguous()
        else:
            partial_max_p = partial_max
            partial_sum_p = partial_sum

        grid2 = (b, c)
        softmax_finalize_kernel[grid2](
            partial_max_p, partial_sum_p,
            global_max, global_inv,
            scale_flat,
            NBLOCKS=NB_PADDED,
            num_warps=1, num_stages=1,
        )

        grid3 = (b, c, NBLOCKS)
        softmax_write_kernel[grid3](
            x_flat, global_max, global_inv, out,
            S,
            clamp_min=self.clamp_min, clamp_max=self.clamp_max,
            BLOCK=BLOCK,
            num_warps=16, num_stages=2,
        )

        return out.view(b, c, d, h, w)