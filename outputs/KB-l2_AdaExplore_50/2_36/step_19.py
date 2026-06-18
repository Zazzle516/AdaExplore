import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 256}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv_transpose_min_sum_gelu_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC: tl.constexpr, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N, OW_tiles)
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_start = pid_w * BLOCK_W
    ow_offs = w_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # Load conv bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    x_base = pid_n * IC * IH * IW

    for oh in range(0, OH):
        conv_tile = b_vals[:, None] + tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            ih_ok = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih >= 0) & (ih < IH)

            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PW - kw
                iw = iw_num // SW
                iw_ok = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw >= 0) & (iw < IW)
                col_valid = iw_ok & ow_mask

                if ih_ok:
                    # Load weight tile [BLOCK_IC, BLOCK_OC] for this (kh, kw)
                    # weight layout: [IC, OC, KH, KW]
                    w_off = (ic_offs[:, None] * (OC * KH * KW)
                             + oc_offs[None, :] * (KH * KW)
                             + kh * KW + kw)
                    w_tile = tl.load(
                        w_ptr + w_off,
                        mask=ic_mask[:, None] & oc_mask[None, :],
                        other=0.0,
                    )  # [BLOCK_IC, BLOCK_OC]

                    # Load x tile [BLOCK_IC, BLOCK_W] at position (ih, iw)
                    x_off = (x_base
                             + ic_offs[:, None] * (IH * IW)
                             + ih * IW
                             + iw[None, :])
                    x_tile = tl.load(
                        x_ptr + x_off,
                        mask=ic_mask[:, None] & col_valid[None, :],
                        other=0.0,
                    )  # [BLOCK_IC, BLOCK_W]

                    # conv_tile [BLOCK_OC, BLOCK_W] += w_tile.T @ x_tile
                    conv_tile += tl.dot(tl.trans(w_tile), x_tile, allow_tf32=False)

        conv_tile = tl.where(oc_mask[:, None], conv_tile, float('inf'))
        min_vec = tl.min(conv_tile, axis=0)
        acc += min_vec

    inv_sqrt2 = 0.7071067811865475
    gelu_out = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    bias_val = tl.load(bias_ptr)
    result = gelu_out + bias_val

    out_offs = pid_n * OW + ow_offs
    tl.store(out_ptr + out_offs, result, mask=ow_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OP = self.output_padding
        OH = (IH - 1) * SH - 2 * PH + KH + OP
        OW = (IW - 1) * SW - 2 * PW + KW + OP

        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias = self.bias.contiguous()

        out = torch.empty((N, 1, 1, OW), device=x.device, dtype=x.dtype)

        # pick BLOCK_OC as next pow2 >= OC
        def next_pow2(n):
            p = 1
            while p < n:
                p *= 2
            return p

        BLOCK_OC = next_pow2(OC)
        BLOCK_IC = next_pow2(IC)

        grid = lambda META: (N, triton.cdiv(OW, META['BLOCK_W']))

        conv_transpose_min_sum_gelu_kernel[grid](
            x, weight, conv_bias, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
        )

        return out