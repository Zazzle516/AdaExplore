import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-add from input side: one program per (N, OC_tile), accumulate scalar per (N,OC).
# Loop over (IC, IH, IW) tiles, multiply by (OC_tile, IC, KH, KW) weight, sum contribution
# of each (kh,kw) by whether input pixel maps to a valid output position.
#
# For each input pixel x[n,ic,ih,iw], its contribution to output[n,oc,oh,ow] is for
# oh = ih*SH - PH + kh, ow = iw*SW - PW + kw, valid if 0<=oh<OH, 0<=ow<OW.
# The contribution to the mean (sum over oh,ow) is x * w[ic,oc,kh,kw] * valid(kh,kw,ih,iw).
#
# We do the full multiply-add count: N * OC * IC * IH * IW * KH * KW (modulo border invalid).


@triton.jit
def convt_mean_scatter_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC, 1, 1] flattened [N*OC]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OC: tl.constexpr,
    BLOCK_IHW: tl.constexpr,
):
    n = tl.program_id(0)
    oc_blk = tl.program_id(1)

    oc_offs = oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    IHW = IH * IW
    inv_norm = multiplier / (OH * OW).to(tl.float32)

    # acc[oc] - accumulator per output channel in this block
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # Iterate over (ic, ih, iw) - the input grid
    num_tiles = (IHW + BLOCK_IHW - 1) // BLOCK_IHW

    for ic in range(0, IC):
        # Preload all KH*KW weights for this (ic, oc_blk) -> [BLOCK_OC, KH*KW]
        # w[ic, oc, kh, kw]
        for t in range(0, num_tiles):
            ihw_offs = t * BLOCK_IHW + tl.arange(0, BLOCK_IHW)
            ihw_mask = ihw_offs < IHW
            ih = ihw_offs // IW
            iw = ihw_offs % IW

            # Load x[n, ic, ih, iw] -> [BLOCK_IHW]
            x_off = n * (IC * IHW) + ic * IHW + ihw_offs
            xv = tl.load(x_ptr + x_off, mask=ihw_mask, other=0.0)

            # For each (kh, kw), compute oh, ow and check validity
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    oh = ih * STRIDE_H - PAD_H + kh
                    ow = iw * STRIDE_W - PAD_W + kw
                    valid = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW) & ihw_mask
                    # sum over ihw of xv * valid -> scalar
                    contrib = tl.sum(tl.where(valid, xv, 0.0), axis=0)

                    # load weight slice w[ic, oc_offs, kh, kw] -> [BLOCK_OC]
                    w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                    wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                    acc += contrib * wv

    # add bias and normalize
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    OHW = OH * OW
    total = (acc + bias * OHW) * inv_norm

    out_off = n * OC + oc_offs
    tl.store(out_ptr + out_off, total, mask=oc_mask)


# The above pre-reduces over (ih,iw) of x before multiplying by w, which violates the
# safety contract (folds the OH/OW reduction into a smaller GEMM). We must instead do
# full per-element multiply.
#
# Approach 2 (safe): one program per (N, OC_tile). Loop over (IC, IH, IW) in tiles. For
# each input pixel, for each (kh,kw), multiply by weight, mask by validity, accumulate.
# Multiply per element happens BEFORE the spatial reduction.


@triton.jit
def convt_mean_kernel_safe(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N*OC]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OC: tl.constexpr,
    BLOCK_IHW: tl.constexpr,
):
    n = tl.program_id(0)
    oc_blk = tl.program_id(1)

    oc_offs = oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    IHW = IH * IW
    OHW = OH * OW
    inv_norm = multiplier / OHW.to(tl.float32)

    # acc[BLOCK_OC]
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    num_tiles = (IHW + BLOCK_IHW - 1) // BLOCK_IHW

    for ic in range(0, IC):
        # preload weights for this ic: [BLOCK_OC, KH*KW]
        # weight layout: w[ic, oc, kh, kw], stride: oc*(KH*KW) + kh*KW + kw
        kh_kw = tl.arange(0, KH * KW)
        w_block_off = ic * (OC * KH * KW) + oc_offs[:, None] * (KH * KW) + kh_kw[None, :]
        w_block_mask = oc_mask[:, None]
        w_block = tl.load(w_ptr + w_block_off, mask=w_block_mask, other=0.0)  # [BLOCK_OC, KH*KW]

        for t in range(0, num_tiles):
            ihw_offs = t * BLOCK_IHW + tl.arange(0, BLOCK_IHW)
            ihw_mask = ihw_offs < IHW
            ih = ihw_offs // IW
            iw = ihw_offs % IW

            x_off = n * (IC * IHW) + ic * IHW + ihw_offs
            xv = tl.load(x_ptr + x_off, mask=ihw_mask, other=0.0)  # [BLOCK_IHW]

            # For each (kh, kw), compute partial sum over ihw, then accumulate
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    oh = ih * STRIDE_H - PAD_H + kh
                    ow = iw * STRIDE_W - PAD_W + kw
                    valid = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW) & ihw_mask
                    # contribution scalar for this (kh,kw): sum over ihw of xv*valid
                    s = tl.sum(tl.where(valid, xv, 0.0), axis=0)
                    # weight slice [BLOCK_OC] for this (kh,kw)
                    w_slice = tl.load(w_ptr + ic * (OC * KH * KW)
                                      + oc_offs * (KH * KW) + kh * KW + kw,
                                      mask=oc_mask, other=0.0)
                    acc += s * w_slice

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    total = (acc + bias * OHW) * inv_norm

    tl.store(out_ptr + n * OC + oc_offs, total, mask=oc_mask)


# The "safe" kernel still reduces x along (ih,iw) per (kh,kw) before multiplying by weight.
# But the safety contract: full asymptotic multiply-add count must be preserved.
# Reference: out[n,oc,oh,ow] = sum_{ic,kh,kw} x*w; mean over (oh,ow).
# Multiply count: N*OC*OH*OW*IC*KH*KW.
# Above approach: per (ic, kh, kw), sums x over ihw then multiplies by single w -> only
# N*OC*IC*KH*KW multiplies. That's NOT compliant.
#
# So we MUST keep x*w as a per-pixel multiply. The compliant kernel:
# acc[oc] += sum_{ic,ih,iw,kh,kw} x[n,ic,ih,iw] * w[ic,oc,kh,kw] * valid(ih,iw,kh,kw)
# We need per-element multiplies x*w then sum. Tile x as [BLOCK_IHW], w as [BLOCK_OC],
# compute outer product [BLOCK_IHW, BLOCK_OC], multiply by valid mask per (kh,kw), accumulate.


@triton.jit
def convt_mean_full_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N*OC]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OC: tl.constexpr,
    BLOCK_IHW: tl.constexpr,
):
    n = tl.program_id(0)
    oc_blk = tl.program_id(1)

    oc_offs = oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    IHW = IH * IW
    OHW = OH * OW
    inv_norm = multiplier / OHW.to(tl.float32)

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    num_tiles = (IHW + BLOCK_IHW - 1) // BLOCK_IHW

    for ic in range(0, IC):
        for t in range(0, num_tiles):
            ihw_offs = t * BLOCK_IHW + tl.arange(0, BLOCK_IHW)
            ihw_mask = ihw_offs < IHW
            ih = ihw_offs // IW
            iw = ihw_offs % IW

            x_off = n * (IC * IHW) + ic * IHW + ihw_offs
            xv = tl.load(x_ptr + x_off, mask=ihw_mask, other=0.0)  # [BLOCK_IHW]

            # For each (kh,kw), multiply per-pixel by weight (per-OC), accumulate
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    oh = ih * STRIDE_H - PAD_H + kh
                    ow = iw * STRIDE_W - PAD_W + kw
                    valid = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW) & ihw_mask

                    # weight [BLOCK_OC] for (ic, oc, kh, kw)
                    w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                    wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    # outer product per-element multiply: [BLOCK_IHW, BLOCK_OC]
                    xv_masked = tl.where(valid, xv, 0.0)
                    prod = xv_masked[:, None] * wv[None, :]
                    # reduce over ihw axis
                    acc += tl.sum(prod, axis=0)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    total = (acc + bias * OHW) * inv_norm

    tl.store(out_ptr + n * OC + oc_offs, total, mask=oc_mask)


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

        BLOCK_OC = 32
        BLOCK_IHW = 256

        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC)
        convt_mean_full_kernel[grid](
            x, w, b, out,
            N, IC, OC, IH, IW, OH, OW,
            KH, KW, SH, SW, PH, PW,
            float(self.multiplier),
            BLOCK_OC=BLOCK_OC,
            BLOCK_IHW=BLOCK_IHW,
            num_warps=4,
            num_stages=2,
        )

        return out