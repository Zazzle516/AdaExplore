import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def instance_norm_div_kernel_large(
    x_ptr, out_ptr,
    HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW

    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.where(mask, x, 0.0)
        sum_sq += tl.where(mask, x * x, 0.0)

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sum_sq, axis=0)
    mean = s / HW
    var = sq / HW - mean * mean
    rstd = tl.rsqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * scale + shift
        tl.store(out_ptr + base + idx, y, mask=mask)


@triton.jit
def instance_norm_div_kernel_single(
    x_ptr, out_ptr,
    HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW

    offs = tl.arange(0, BLOCK)
    mask = offs < HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    xm = tl.where(mask, xf, 0.0)
    s = tl.sum(xm, axis=0)
    sq = tl.sum(xm * xm, axis=0)
    mean = s / HW
    var = sq / HW - mean * mean
    rstd = tl.rsqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale
    y = xf * scale + shift
    tl.store(out_ptr + base + offs, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = float(divide_by)
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        HW = H * W
        x_contig = x.contiguous()
        out = torch.empty_like(x_contig)

        grid = (N * C,)
        inv_div = 1.0 / self.divide_by

        def next_pow2(n):
            p = 1
            while p < n:
                p <<= 1
            return p

        if HW <= 32768:
            BLOCK = next_pow2(HW)
            if BLOCK < 64:
                BLOCK = 64
            if BLOCK <= 1024:
                num_warps = 4
            elif BLOCK <= 4096:
                num_warps = 8
            else:
                num_warps = 16
            instance_norm_div_kernel_single[grid](
                x_contig, out,
                HW,
                inv_div,
                self.eps,
                BLOCK=BLOCK,
                num_warps=num_warps,
                num_stages=2,
            )
        else:
            BLOCK = 4096
            instance_norm_div_kernel_large[grid](
                x_contig, out,
                HW,
                inv_div,
                self.eps,
                BLOCK=BLOCK,
                num_warps=8,
                num_stages=2,
            )
        return out