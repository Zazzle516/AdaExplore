import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, D, H, W,
    PD, PH, PW,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,  # pool window volume (kd*kh*kw)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
):
    pid = tl.program_id(0)
    # decode pid -> (n, pd, ph, pw)
    pw = pid % PW
    tmp = pid // PW
    ph = tmp % PH
    tmp2 = tmp // PH
    pd = tmp2 % PD
    n = tmp2 // PD

    # pool start coords (stride = kernel size, padding=0)
    d0 = pd * KD
    h0 = ph * KH
    w0 = pw * KW

    offs_c = tl.arange(0, BLOCK_C)[:, None]  # [BLOCK_C, 1]
    mask_c = offs_c < C
    offs_k = tl.arange(0, BLOCK_K)[None, :]  # [1, BLOCK_K]

    # decompose offs_k into (kd, kh, kw)
    kd = offs_k // (KH * KW)
    rem = offs_k % (KH * KW)
    kh = rem // KW
    kw = rem % KW
    mask_k = offs_k < (KD * KH * KW)

    dd = d0 + kd
    hh = h0 + kh
    ww = w0 + kw

    # base offset per (c, k)
    # x layout: [N, C, D, H, W]
    base_n = n * C * D * H * W
    ptr = x_ptr + base_n + offs_c * (D * H * W) + dd * (H * W) + hh * W + ww
    mask = mask_c & mask_k

    vals = tl.load(ptr, mask=mask, other=-float('inf'))
    # max over pool window (axis=1) per channel
    pooled = tl.max(vals, axis=1)  # [BLOCK_C]
    mask_c_1d = tl.arange(0, BLOCK_C) < C
    pooled = tl.where(mask_c_1d, pooled, -float('inf'))

    # softmax across channels
    x_max = tl.max(pooled, axis=0)
    e = tl.exp(pooled - x_max)
    e = tl.where(mask_c_1d, e, 0.0)
    denom = tl.sum(e, axis=0)
    sm = e / denom

    sub = tl.load(sub_ptr + tl.arange(0, BLOCK_C), mask=mask_c_1d, other=0.0)
    y = sm - sub

    sig = 1.0 / (1.0 + tl.exp(-y))
    sw = sig * y

    sw = tl.where(mask_c_1d, sw, -float('inf'))
    res = tl.max(sw, axis=0)

    out_idx = ((n * PD + pd) * PH + ph) * PW + pw
    tl.store(out_ptr + out_idx, res)


def fused_post(x, sub, pool_k=2):
    # x: conv output [N, C, D, H, W]
    N, C, D, H, W = x.shape
    KD = KH = KW = pool_k
    PD = D // KD
    PH = H // KH
    PW = W // KW
    x = x.contiguous()
    out = torch.empty((N, PD, PH, PW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16
    BLOCK_K = triton.next_power_of_2(KD * KH * KW)
    if BLOCK_K < 8:
        BLOCK_K = 8

    grid = (N * PD * PH * PW,)
    fused_pool_softmax_sub_swish_max_kernel[grid](
        x, sub, out,
        N, C, D, H, W,
        PD, PH, PW,
        BLOCK_C=BLOCK_C,
        BLOCK_K=BLOCK_K,
        KD=KD, KH=KH, KW=KW,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_post(x, self.subtract, pool_k=2)
        return x