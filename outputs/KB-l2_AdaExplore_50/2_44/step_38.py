import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Key insight: output is mean over H,W of conv_transpose(x) * multiplier.
# By linearity: mean_{oh,ow} sum_{ic,kh,kw} x[n,ic,ih,iw] * w[ic,oc,kh,kw]
# = sum_{ic,kh,kw} w[ic,oc,kh,kw] * (1/(OH*OW)) * sum_{oh,ow valid} x[n,ic,ih(oh,kh),iw(ow,kw)]
#
# But the safety contract forbids pre-reducing along axes a downstream reduction will collapse.
# So we must materialize the full output shape and perform the same asymptotic work.
#
# Strategy: one program per (n, oc), each iterates over the full OH*OW grid in tiles.
# For each tile, compute the conv_transpose output values (full work) and accumulate sum.
# This still does N*OC*OH*OW*IC*KH*KW work (same as reference).


@triton.jit
def fused_convt_mean_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC, 1, 1]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OHW: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)

    OHW = OH * OW
    inv_norm = multiplier / (OH * OW).to(tl.float32)

    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over output spatial in tiles
    num_tiles = (OHW + BLOCK_OHW - 1) // BLOCK_OHW

    for t in range(0, num_tiles):
        offs = t * BLOCK_OHW + tl.arange(0, BLOCK_OHW)
        mask_ohw = offs < OHW
        oh = offs // OW
        ow = offs % OW

        tile_sum = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

        # loop over kh, kw, ic
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                num_h = oh + PAD_H - kh
                num_w = ow + PAD_W - kw
                ih = num_h // STRIDE_H
                iw = num_w // STRIDE_W
                valid = (num_h % STRIDE_H == 0) & (num_w % STRIDE_W == 0) \
                        & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & mask_ohw

                # Load weight slice w[:, oc, kh, kw] across all IC
                # then accumulate sum_ic x[n,ic,ih,iw] * w[ic,oc,kh,kw]
                for ic in range(0, IC):
                    w_off = ic * (OC * KH * KW) + oc * (KH * KW) + kh * KW + kw
                    wv = tl.load(w_ptr + w_off)
                    x_off = n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                    xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    tile_sum += xv * wv

        acc += tl.sum(tl.where(mask_ohw, tile_sum, 0.0), axis=0)

    bias = tl.load(b_ptr + oc)
    total = (acc + bias * OHW) * inv_norm
    tl.store(out_ptr + n * OC + oc, total)


# Better approach: tile over IC too in a register-resident manner using tl.dot-style
# Actually the inner loop above does IC iterations of scalar work per output - inefficient.
# Let's use a different layout: load BLOCK_IC weights and BLOCK_OHW * BLOCK_IC x values at once.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OHW': 512, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OHW': 512, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OHW': 1024, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OHW': 1024, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OHW': 1024, 'BLOCK_IC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OHW': 2048, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OHW': 2048, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OHW': 1024, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def fused_convt_mean_kernel_v2(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC, 1, 1]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OHW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    n = tl.program_id(0)
    oc = tl.program_id(1)

    OHW = OH * OW
    inv_norm = multiplier / OHW.to(tl.float32)

    acc = tl.zeros((), dtype=tl.float32)

    num_tiles = (OHW + BLOCK_OHW - 1) // BLOCK_OHW

    x_base_n = n * IC * IH * IW
    w_base_oc = oc * KH * KW

    for t in range(0, num_tiles):
        offs = t * BLOCK_OHW + tl.arange(0, BLOCK_OHW)
        mask_ohw = offs < OHW
        oh = offs // OW
        ow = offs % OW

        tile_sum = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            num_h = oh + PAD_H - kh
            ih = num_h // STRIDE_H
            valid_h = (num_h % STRIDE_H == 0) & (ih >= 0) & (ih < IH)
            ih_IW = ih * IW

            for kw in tl.static_range(0, KW):
                num_w = ow + PAD_W - kw
                iw = num_w // STRIDE_W
                valid = valid_h & (num_w % STRIDE_W == 0) & (iw >= 0) & (iw < IW) & mask_ohw

                spatial_off = ih_IW + iw  # [BLOCK_OHW]

                # Tile over IC
                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    w_off = ic_offs * (OC * KH * KW) + w_base_oc + kh * KW + kw
                    wv = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)

                    x_off = (x_base_n
                             + ic_offs[None, :] * (IH * IW)
                             + spatial_off[:, None])
                    x_mask = valid[:, None] & ic_mask[None, :]
                    xv = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    tile_sum += tl.sum(xv * wv[None, :], axis=1)

        acc += tl.sum(tl.where(mask_ohw, tile_sum, 0.0), axis=0)

    bias = tl.load(b_ptr + oc)
    total = (acc + bias * OHW) * inv_norm
    tl.store(out_ptr + n * OC + oc, total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous().cuda()
        b = self.conv_transpose.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding

        OH = (IH - 1) * SH - 2 * PH + KH + self.output_padding
        OW = (IW - 1) * SW - 2 * PW + KW + self.output_padding

        out = torch.empty((N, OC, 1, 1), device=x.device, dtype=torch.float32)

        grid = (N, OC)
        fused_convt_mean_kernel_v2[grid](
            x, w, b, out,
            N, IC, OC, IH, IW, OH, OW,
            KH, KW, SH, SW, PH, PW,
            float(self.multiplier),
        )

        return out