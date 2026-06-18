import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _conv_mish_bn_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)  # output spatial tile (over OH*OW per (n))
    pid_n = tl.program_id(1)  # OC tile
    pid_b = tl.program_id(2)  # batch index

    OHW = OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial positions within batch
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # OC

    oh = offs_m // OW
    ow = offs_m % OW

    K = IC * KH * KW
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k  # [BLOCK_K]
        ic = k // (KH * KW)
        rem = k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # input addresses: x[pid_b, ic, oh+kh, ow+kw]
        ih = oh[:, None] + kh[None, :]   # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]
        x_off = pid_b * IC * IH * IW + ic[None, :] * IH * IW + ih * IW + iw
        m_mask = (offs_m[:, None] < OHW) & (k[None, :] < K)
        x_vals = tl.load(x_ptr + x_off, mask=m_mask, other=0.0)

        # weight: w[oc, ic, kh, kw] -> w_ptr + oc*IC*KH*KW + k
        w_off = offs_n[None, :] * K + k[:, None]
        w_mask = (offs_n[None, :] < OC) & (k[:, None] < K)
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + bias[None, :]

    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(acc))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = acc * th

    # bn affine
    s = tl.load(scale_ptr + offs_n, mask=offs_n < OC, other=0.0)
    b_ = tl.load(shift_ptr + offs_n, mask=offs_n < OC, other=0.0)
    out = y * s[None, :] + b_[None, :]

    # store: out[pid_b, oc, oh, ow]
    out_off = pid_b * OC * OHW + offs_n[None, :] * OHW + offs_m[:, None]
    out_mask = (offs_m[:, None] < OHW) & (offs_n[None, :] < OC)
    tl.store(out_ptr + out_off, out, mask=out_mask)


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
    sp = tl.log(1.0 + tl.exp(x))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = x * th

    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    b = tl.load(shift_ptr + c, mask=mask, other=0.0)
    out = y * s + b
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _mish_stats_kernel(
    x_ptr, y_ptr, sum_ptr, sqsum_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
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
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.training:
            x = self.conv(x)
            N, C, H, W = x.shape
            HW = H * W
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
            # Eval path: fused conv + mish + BN affine
            N, IC, IH, IW = x.shape
            OC = self.out_channels
            KH = self.kernel_size
            KW = self.kernel_size
            OH = IH - KH + 1
            OW = IW - KW + 1

            invstd = torch.rsqrt(self.bn.running_var + self.eps)
            scale = (self.bn.weight * invstd).contiguous()
            shift = (self.bn.bias - self.bn.running_mean * scale).contiguous()

            x = x.contiguous()
            w = self.conv.weight.contiguous()
            b = self.conv.bias.contiguous()

            out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

            BLOCK_M = 64
            BLOCK_N = 64
            BLOCK_K = 32

            grid = (
                (OH * OW + BLOCK_M - 1) // BLOCK_M,
                (OC + BLOCK_N - 1) // BLOCK_N,
                N,
            )
            _conv_mish_bn_kernel[grid](
                x, w, b, scale, shift, out,
                N, IC, IH, IW,
                OC, OH, OW,
                KH, KW,
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=2,
            )
            return out