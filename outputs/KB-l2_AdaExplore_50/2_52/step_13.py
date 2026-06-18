import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _mish_bn_kernel(
    x_ptr, out_ptr, scale_ptr, shift_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * HW
    mask = offs < total
    
    c = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(x))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = x * th
    
    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    b = tl.load(shift_ptr + c, mask=mask, other=0.0)
    out = y * s + b
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _mish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sp = tl.log(1.0 + tl.exp(x))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = x * th
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def _mish_stats_kernel(
    x_ptr, y_ptr, sum_ptr, sqsum_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    # one program per (n, c): compute mish and accumulate sum/sqsum
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    base = n * C * HW + c * HW
    offs = tl.arange(0, BLOCK)
    
    acc_sum = tl.zeros([BLOCK], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK], dtype=tl.float32)
    
    for start in range(0, HW, BLOCK):
        idx = start + offs
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sp = tl.log(1.0 + tl.exp(x))
        e2 = tl.exp(2.0 * sp)
        th = (e2 - 1.0) / (e2 + 1.0)
        y = x * th
        tl.store(y_ptr + base + idx, y, mask=mask)
        acc_sum += tl.where(mask, y, 0.0)
        acc_sq += tl.where(mask, y * y, 0.0)
    
    s = tl.sum(acc_sum, axis=0)
    sq = tl.sum(acc_sq, axis=0)
    
    tl.atomic_add(sum_ptr + c, s)
    tl.atomic_add(sqsum_ptr + c, sq)


@triton.jit
def _stats_kernel(
    y_ptr, sum_ptr, sqsum_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    # one program per (n, c) computes partial sum/sqsum across HW
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    base = n * C * HW + c * HW
    offs = tl.arange(0, BLOCK)
    
    acc_sum = tl.zeros([BLOCK], dtype=tl.float32)
    acc_sq = tl.zeros([BLOCK], dtype=tl.float32)
    
    for start in range(0, HW, BLOCK):
        idx = start + offs
        mask = idx < HW
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc_sum += tl.where(mask, v, 0.0)
        acc_sq += tl.where(mask, v * v, 0.0)
    
    s = tl.sum(acc_sum, axis=0)
    sq = tl.sum(acc_sq, axis=0)
    
    # atomic add into per-channel buffer
    tl.atomic_add(sum_ptr + c, s)
    tl.atomic_add(sqsum_ptr + c, sq)


@triton.jit
def _apply_bn_kernel(
    y_ptr, out_ptr, scale_ptr, shift_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * HW
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
            # fused mish + stats
            y = torch.empty_like(x)
            sum_buf = torch.zeros(C, device=x.device, dtype=torch.float32)
            sqsum_buf = torch.zeros(C, device=x.device, dtype=torch.float32)
            stats_grid = (N * C,)
            _mish_stats_kernel[stats_grid](x, y, sum_buf, sqsum_buf, N, C, HW, BLOCK=2048, num_warps=8)
            
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
            BLOCK = 4096
            grid2 = ((n_el + BLOCK - 1) // BLOCK,)
            _apply_bn_kernel[grid2](y, out, scale, shift, N, C, HW, BLOCK=BLOCK, num_warps=8)
            return out
        else:
            invstd = torch.rsqrt(self.bn.running_var + self.eps)
            scale = self.bn.weight * invstd
            shift = self.bn.bias - self.bn.running_mean * scale
            
            out = torch.empty_like(x)
            n_el = x.numel()
            BLOCK = 4096
            grid = ((n_el + BLOCK - 1) // BLOCK,)
            _mish_bn_kernel[grid](x, out, scale, shift, N, C, HW, BLOCK=BLOCK, num_warps=8)
            return out