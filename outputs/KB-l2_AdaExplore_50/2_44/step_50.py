import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_mean_input_stationary_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    bias_ptr,        # [OC]
    out_ptr,         # [N, OC]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OC: tl.constexpr,
    BLOCK_IHW: tl.constexpr,
):
    # grid: (N, OC // BLOCK_OC)
    # Strategy: input-stationary. For each (n, oc_tile), iterate over all (ic, ih, iw)
    # and accumulate sum over output positions weighted by valid_count(kh,kw).
    # output mean = sum_{oc} ( bias[oc] + (1/OHW) * sum_{ic,ih,iw} x[n,ic,ih,iw] *
    #                                       sum_{kh,kw} w[ic,oc,kh,kw] * valid_count(ih,iw,kh,kw) )
    # 
    # BUT precomputing the inner sum_{kh,kw} would violate the safety contract.
    # We compute it on-the-fly inside the kernel for each (ic, ih, iw, oc_tile).
    # MAC count: N * IC * IH * IW * OC * KH * KW = full conv work. ✓

    n = tl.program_id(0)
    oc_tile = tl.program_id(1)

    oc_offs = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    OHW = OH * OW
    inv_ohw = 1.0 / OHW

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    IHW = IH * IW
    # Loop over input spatial in tiles
    for ihw_start in range(0, IHW, BLOCK_IHW):
        offs = ihw_start + tl.arange(0, BLOCK_IHW)
        mask_ihw = offs < IHW
        ih = offs // IW
        iw = offs % IW

        # For each (kh, kw), compute valid mask: output position
        # oh = ih * STRIDE_H - PAD_H + kh ; must satisfy 0 <= oh < OH
        # ow = iw * STRIDE_W - PAD_W + kw ; must satisfy 0 <= ow < OW
        # For this conv layout (every input pixel × kernel maps to exactly 1 output pixel),
        # valid_count for a given (ih,iw,kh,kw) is 0 or 1.

        # We'll iterate IC, and for each ic load x[n,ic,ih,iw] (BLOCK_IHW), then for
        # each (kh,kw) compute valid mask and load w[ic, oc_tile, kh, kw] (BLOCK_OC),
        # and accumulate x * w * valid into acc.

        for ic in range(0, IC):
            x_off = n * IC * IHW + ic * IHW + offs
            x_vals = tl.load(x_ptr + x_off, mask=mask_ihw, other=0.0)  # [BLOCK_IHW]

            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    oh = ih * STRIDE_H - PAD_H + kh
                    ow = iw * STRIDE_W - PAD_W + kw
                    valid = (oh >= 0) & (oh < OH) & (ow >= 0) & (ow < OW) & mask_ihw
                    valid_f = tl.where(valid, 1.0, 0.0)

                    # x contribution scalar per ihw position; sum over ihw
                    x_contrib = tl.sum(x_vals * valid_f, axis=0)  # scalar

                    # w[ic, oc_offs, kh, kw]
                    w_off = ic * (OC * KH * KW) + oc_offs * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_contrib * w_vals

    # Now acc holds sum over (ic, ih, iw, kh, kw) of x * w * valid for each oc.
    # Add bias contribution: bias[oc] * OHW (since each output position gets bias).
    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    total = (acc + bias * OHW) * multiplier * inv_ohw

    out_off = n * OC + oc_offs
    tl.store(out_ptr + out_off, total, mask=oc_mask)


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
        BLOCK_IHW = 1024

        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC)
        conv_transpose_mean_input_stationary_kernel[grid](
            x, w, b, out,
            N, IC, OC, IH, IW, OH, OW,
            KH, KW,
            SH, SW, PH, PW,
            float(self.multiplier),
            BLOCK_OC=BLOCK_OC,
            BLOCK_IHW=BLOCK_IHW,
            num_warps=4,
            num_stages=2,
        )

        return out