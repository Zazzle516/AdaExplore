import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _mish(x):
    sp = tl.log(1.0 + tl.exp(x))
    e1 = tl.exp(sp)
    e2 = tl.exp(-sp)
    t = (e1 - e2) / (e1 + e2)
    return x * t


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['N', 'OC', 'IC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv2d_mish_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    # x strides (NHWC): n, h, w, c
    sx_n, sx_h, sx_w, sx_c,
    # w strides (OC, IC, KH, KW): oc, ic, kh, kw
    sw_oc, sw_ic, sw_kh, sw_kw,
    # out strides (NHWC): n, h, w, c
    so_n, so_h, so_w, so_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)        # batch
    pid_oc = tl.program_id(1)       # OC tile
    pid_m = tl.program_id(2)        # OH*OW tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # output spatial idx
    offs_oc = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)      # output channels

    oh = offs_m // OW
    ow = offs_m % OW
    m_mask = offs_m < (OH * OW)
    n_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K dim = IC * KH * KW. For 3x3, only 9 (kh,kw); inner-most is IC contiguous.
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # padding=0
            iw = ow + kw
            for ic_start in range(0, IC, BLOCK_K):
                offs_ic = ic_start + tl.arange(0, BLOCK_K)
                ic_mask = offs_ic < IC

                # Load x[n, ih, iw, ic] : shape (BLOCK_M, BLOCK_K)
                x_off = (pid_n * sx_n
                         + ih[:, None] * sx_h
                         + iw[:, None] * sx_w
                         + offs_ic[None, :] * sx_c)
                x_mask = m_mask[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # Load w[oc, ic, kh, kw] : shape (BLOCK_K, BLOCK_N)
                w_off = (offs_oc[None, :] * sw_oc
                         + offs_ic[:, None] * sw_ic
                         + kh * sw_kh
                         + kw * sw_kw)
                w_mask = ic_mask[:, None] & n_mask[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # bias
    bias = tl.load(b_ptr + offs_oc, mask=n_mask, other=0.0)
    acc += bias[None, :]

    # double mish
    acc = _mish(acc)
    acc = _mish(acc)

    # store NHWC
    out_off = (pid_n * so_n
               + oh[:, None] * so_h
               + ow[:, None] * so_w
               + offs_oc[None, :] * so_c)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def conv2d_mish_mish(x, weight, bias):
    # x: (N, IC, H, W)  -> NHWC
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sw_oc, sw_ic, sw_kh, sw_kw = weight.stride()
    so_n, so_h, so_w, so_c = out_nhwc.stride()

    grid = lambda META: (
        N,
        triton.cdiv(OC, META['BLOCK_N']),
        triton.cdiv(OH * OW, META['BLOCK_M']),
    )

    conv2d_mish_mish_kernel[grid](
        x_nhwc, weight, bias, out_nhwc,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        sx_n, sx_h, sx_w, sx_c,
        sw_oc, sw_ic, sw_kh, sw_kw,
        so_n, so_h, so_w, so_c,
    )

    out = out_nhwc.permute(0, 3, 1, 2).contiguous()
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv2d_mish_mish(x, self.conv.weight, self.conv.bias)