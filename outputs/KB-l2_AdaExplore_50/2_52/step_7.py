import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def mish_kernel(
    x_ptr, out_ptr,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    e2 = tl.exp(2.0 * sp)
    t = (e2 - 1.0) / (e2 + 1.0)
    y = x * t
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def mish_bn_eval_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    C, HW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = (offs // HW) % C
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    e2 = tl.exp(2.0 * sp)
    t = (e2 - 1.0) / (e2 + 1.0)
    mish = x * t
    y = mish * scale + shift
    tl.store(out_ptr + offs, y, mask=mask)


@triton.jit
def bn_reduce_kernel(
    x_ptr,
    sum_ptr, sumsq_ptr,
    N, C, HW,
    NHW,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per channel
    c = tl.program_id(0)
    sum_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # iterate over N*HW elements for channel c
    # element index across (n, hw) is n*C*HW + c*HW + hw
    for off in range(0, NHW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < NHW
        n = idx // HW
        hw = idx % HW
        ptr = x_ptr + n * (C * HW) + c * HW + hw
        v = tl.load(ptr, mask=mask, other=0.0)
        sum_acc += tl.where(mask, v, 0.0)
        sumsq_acc += tl.where(mask, v * v, 0.0)

    s = tl.sum(sum_acc, axis=0)
    sq = tl.sum(sumsq_acc, axis=0)
    tl.store(sum_ptr + c, s)
    tl.store(sumsq_ptr + c, sq)


@triton.jit
def bn_apply_train_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    C, HW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = (offs // HW) % C
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)
    y = x * scale + shift
    tl.store(out_ptr + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KHKW'],
)
@triton.jit
def conv2d_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    OUT_HW, IC_KHKW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    APPLY_MISH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, IC_KHKW, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < IC_KHKW

        ic = offs_k // (KH * KW)
        khw = offs_k % (KH * KW)
        kh = khw // KW
        kw = khw % KW

        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]

        x_ptrs = (x_ptr
                  + pid_batch * stride_xn
                  + ic[None, :] * stride_xc
                  + ih * stride_xh
                  + iw * stride_xw)

        m_mask = offs_m < OUT_HW
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = (w_ptr
                  + offs_n[None, :] * stride_wo
                  + ic[:, None] * stride_wi
                  + kh[:, None] * stride_wh
                  + kw[:, None] * stride_ww)
        n_mask = offs_n < OC
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    n_mask = offs_n < OC
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b[None, :]

    if APPLY_MISH:
        sp = tl.where(acc > 20.0, acc, tl.log(1.0 + tl.exp(acc)))
        e2 = tl.exp(2.0 * sp)
        t = (e2 - 1.0) / (e2 + 1.0)
        acc = acc * t

    m_mask = offs_m < OUT_HW
    out_ptrs = (out_ptr
                + pid_batch * stride_on
                + offs_n[None, :] * stride_oc
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_conv2d_mish(x, w, b, apply_mish=True):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    OUT_HW = OH * OW
    IC_KHKW = IC * KH * KW

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(OUT_HW, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
        N,
    )

    conv2d_mish_kernel[grid](
        x, w, b, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        OUT_HW, IC_KHKW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        APPLY_MISH=apply_mish,
    )
    return out


def bn_train_apply(x, bn_weight, bn_bias, running_mean, running_var, momentum, eps):
    N, C, H, W = x.shape
    HW = H * W
    NHW = N * HW
    total = x.numel()

    sum_buf = torch.empty(C, device=x.device, dtype=torch.float32)
    sumsq_buf = torch.empty(C, device=x.device, dtype=torch.float32)

    BLOCK_RED = 1024
    bn_reduce_kernel[(C,)](
        x, sum_buf, sumsq_buf,
        N, C, HW, NHW,
        BLOCK_SIZE=BLOCK_RED,
    )

    mean = sum_buf / NHW
    var = sumsq_buf / NHW - mean * mean
    # unbiased var for running stats
    unbiased_var = var * (NHW / (NHW - 1)) if NHW > 1 else var

    # update running stats in-place
    with torch.no_grad():
        running_mean.mul_(1 - momentum).add_(mean, alpha=momentum)
        running_var.mul_(1 - momentum).add_(unbiased_var, alpha=momentum)

    invstd = torch.rsqrt(var + eps)
    scale = bn_weight * invstd
    shift = bn_bias - mean * scale

    out = torch.empty_like(x)
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    bn_apply_train_kernel[grid](
        x, out, scale.contiguous(), shift.contiguous(),
        C, HW, total, BLOCK_SIZE=BLOCK,
    )
    return out


def mish_bn_eval_apply(x, scale, shift):
    N, C, H, W = x.shape
    HW = H * W
    total = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    mish_bn_eval_kernel[grid](
        x, out, scale, shift,
        C, HW, total, BLOCK_SIZE=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps
        self.momentum = momentum

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(
            self.conv.out_channels, device=x.device, dtype=x.dtype)

        if self.training:
            # fused conv + mish, then BN training
            conv_mish = triton_conv2d_mish(x, w, b, apply_mish=True)
            y = bn_train_apply(
                conv_mish,
                self.bn.weight, self.bn.bias,
                self.bn.running_mean, self.bn.running_var,
                self.momentum, self.eps,
            )
            return y
        else:
            # eval: fuse conv + (mish + bn fold)
            conv_out = triton_conv2d_mish(x, w, b, apply_mish=False)
            rm = self.bn.running_mean
            rv = self.bn.running_var
            bw = self.bn.weight
            bb = self.bn.bias
            invstd = torch.rsqrt(rv + self.eps)
            scale = bw * invstd
            shift = bb - rm * scale
            return mish_bn_eval_apply(conv_out, scale.contiguous(), shift.contiguous())