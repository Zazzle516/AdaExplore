import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def _mish_bn_kernel(
    x_ptr, out_ptr, scale_ptr, shift_ptr,
    total, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    c = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    th = libdevice.tanh(sp)
    y = x * th

    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    b = tl.load(shift_ptr + c, mask=mask, other=0.0)
    out = y * s + b
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 4096}, num_warps=8),
    ],
    key=['NHW'],
)
@triton.jit
def _mish_stats_per_channel_kernel(
    x_ptr, y_ptr, sum_ptr, sqsum_ptr,
    N, C, HW, NHW,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK)

    acc_sum = tl.zeros([BLOCK], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for start in range(0, NHW, BLOCK):
        idx = start + offs
        mask = idx < NHW
        n = idx // HW
        hw = idx % HW
        addr = n * C * HW + c * HW + hw
        x = tl.load(x_ptr + addr, mask=mask, other=0.0)
        sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
        th = libdevice.tanh(sp)
        y = x * th
        tl.store(y_ptr + addr, y, mask=mask)
        acc_sum += tl.where(mask, y, 0.0)
        acc_sq += tl.where(mask, y * y, 0.0)

    s = tl.sum(acc_sum, axis=0)
    sq = tl.sum(acc_sq, axis=0)

    tl.store(sum_ptr + c, s)
    tl.store(sqsum_ptr + c, sq)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def _apply_bn_kernel(
    y_ptr, out_ptr, scale_ptr, shift_ptr,
    total, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    c = (offs // HW) % C
    y = tl.load(y_ptr + offs, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    b = tl.load(shift_ptr + c, mask=mask, other=0.0)
    out = y * s + b
    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        HW = H * W

        if self.training:
            y = torch.empty_like(x)
            sum_buf = torch.empty(C, device=x.device, dtype=torch.float32)
            sqsum_buf = torch.empty(C, device=x.device, dtype=torch.float32)
            NHW = N * HW
            _mish_stats_per_channel_kernel[(C,)](x, y, sum_buf, sqsum_buf, N, C, HW, NHW)

            count = N * HW
            mean = sum_buf / count
            var = sqsum_buf / count - mean * mean
            unbiased_var = var * (count / (count - 1))

            with torch.no_grad():
                self.bn.running_mean.mul_(1 - self.momentum).add_(mean.detach() * self.momentum)
                self.bn.running_var.mul_(1 - self.momentum).add_(unbiased_var.detach() * self.momentum)
                self.bn.num_batches_tracked.add_(1)

            invstd = torch.rsqrt(var + self.eps)
            scale = self.bn.weight * invstd
            shift = self.bn.bias - mean * scale

            out = torch.empty_like(y)
            n_el = x.numel()
            grid2 = lambda meta: ((n_el + meta['BLOCK'] - 1) // meta['BLOCK'],)
            _apply_bn_kernel[grid2](y, out, scale, shift, n_el, C, HW)
            return out
        else:
            invstd = torch.rsqrt(self.bn.running_var + self.eps)
            scale = (self.bn.weight * invstd).contiguous()
            shift = (self.bn.bias - self.bn.running_mean * scale).contiguous()

            out = torch.empty_like(x)
            n_el = x.numel()
            grid = lambda meta: ((n_el + meta['BLOCK'] - 1) // meta['BLOCK'],)
            _mish_bn_kernel[grid](x, out, scale, shift, n_el, C, HW)
            return out