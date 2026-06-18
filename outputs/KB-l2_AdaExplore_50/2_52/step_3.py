import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def mish_bn_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, HW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # compute channel index
    c = (offs // HW) % C

    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)

    # softplus(x) = log(1 + exp(x)); stable
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    t = (e2 - 1.0) / (e2 + 1.0)
    mish = x * t

    y = mish * scale + shift
    tl.store(out_ptr + offs, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KHKW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    OUT_HW, IC_KHKW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_on, stride_oc, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M dim = N * OH * OW, N dim = OC, K dim = IC * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)  # batch index

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial output positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels

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

        # input pointers: x[pid_batch, ic, oh+kh, ow+kw]
        ih = oh[:, None] + kh[None, :]
        iw = ow[:, None] + kw[None, :]

        x_ptrs = (x_ptr
                  + pid_batch * stride_xn
                  + ic[None, :] * stride_xc
                  + ih * stride_xh
                  + iw * stride_xw)

        m_mask = offs_m < OUT_HW
        x_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_M, BLOCK_K]

        # weight: w[offs_n, ic, kh, kw]
        w_ptrs = (w_ptr
                  + offs_n[None, :] * stride_wo
                  + ic[:, None] * stride_wi
                  + kh[:, None] * stride_wh
                  + kw[:, None] * stride_ww)
        n_mask = offs_n < OC
        w_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(x_vals, w_vals)

    # bias
    n_mask = offs_n < OC
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc += b[None, :]

    # store
    m_mask = offs_m < OUT_HW
    out_ptrs = (out_ptr
                + pid_batch * stride_on
                + offs_n[None, :] * stride_oc
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_conv2d(x, w, b):
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

    conv2d_kernel[grid](
        x, w, b, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        OUT_HW, IC_KHKW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


def mish_bn_apply(x, scale, shift):
    N, C, H, W = x.shape
    HW = H * W
    total = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    mish_bn_kernel[grid](
        x, out, scale, shift,
        N, C, HW, total,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, eps=1e-5, momentum=0.1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.eps = eps

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(
            self.conv.out_channels, device=x.device, dtype=x.dtype)

        if self.training:
            # fall back to standard path during training (BN needs running stats updates)
            y = F.conv2d(x, w, b)
            y = torch.multiply(torch.tanh(F.softplus(y)), y)
            y = self.bn(y)
            return y

        # eval: fuse mish + BN
        conv_out = triton_conv2d(x, w, b)

        rm = self.bn.running_mean
        rv = self.bn.running_var
        bw = self.bn.weight
        bb = self.bn.bias
        invstd = torch.rsqrt(rv + self.eps)
        scale = bw * invstd
        shift = bb - rm * scale

        return mish_bn_apply(conv_out, scale.contiguous(), shift.contiguous())