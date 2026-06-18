import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_mean_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    bias_ptr,        # [OC]
    out_ptr,         # [N, OC, 1, 1]
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OHW: tl.constexpr,
):
    # grid: (N, OC)
    n = tl.program_id(0)
    oc = tl.program_id(1)

    OHW = OH * OW
    acc = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

    # We iterate over output positions in tiles of BLOCK_OHW.
    # For each tile, compute the contribution from all (ic, kh, kw).
    num_tiles = (OHW + BLOCK_OHW - 1) // BLOCK_OHW

    for t in range(0, num_tiles):
        offs = t * BLOCK_OHW + tl.arange(0, BLOCK_OHW)
        mask_ohw = offs < OHW
        oh = offs // OW
        ow = offs % OW

        tile_acc = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                num_h = oh + PAD_H - kh
                num_w = ow + PAD_W - kw
                ih = num_h // STRIDE_H
                iw = num_w // STRIDE_W
                valid_h = (num_h % STRIDE_H == 0) & (ih >= 0) & (ih < IH)
                valid_w = (num_w % STRIDE_W == 0) & (iw >= 0) & (iw < IW)
                valid = valid_h & valid_w & mask_ohw

                # Loop over IC
                for ic in range(0, IC):
                    w_off = ic * (OC * KH * KW) + oc * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)

                    x_off = n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    tile_acc += x_val * w_val

        acc += tl.where(mask_ohw, tile_acc, 0.0)

    s = tl.sum(acc, axis=0)
    bias = tl.load(bias_ptr + oc)
    total = (s + bias * OHW) * multiplier / OHW
    tl.store(out_ptr + n * OC + oc, total)


@triton.jit
def conv_transpose_mean_kernel_v2(
    x_ptr,           # [N, IC, IH*IW]
    w_ptr,           # [IC, OC, KH, KW] -> we use slice w[:, oc, :, :] shape [IC, KH*KW]
    bias_ptr,
    out_ptr,
    N, IC, OC, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier,
    BLOCK_OHW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N, OC)
    n = tl.program_id(0)
    oc = tl.program_id(1)

    OHW = OH * OW
    KHW = KH * KW
    IHW = IH * IW

    acc_scalar = tl.zeros((1,), dtype=tl.float32)

    num_tiles = (OHW + BLOCK_OHW - 1) // BLOCK_OHW

    for t in range(0, num_tiles):
        offs = t * BLOCK_OHW + tl.arange(0, BLOCK_OHW)
        mask_ohw = offs < OHW
        oh = offs // OW
        ow = offs % OW

        tile_acc = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                num_h = oh + PAD_H - kh
                num_w = ow + PAD_W - kw
                ih = num_h // STRIDE_H
                iw = num_w // STRIDE_W
                valid_h = (num_h % STRIDE_H == 0) & (ih >= 0) & (ih < IH)
                valid_w = (num_w % STRIDE_W == 0) & (iw >= 0) & (iw < IW)
                valid = valid_h & valid_w & mask_ohw

                for ic_start in range(0, IC, BLOCK_IC):
                    ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                    ic_mask = ic_offs < IC

                    # weight: w[ic, oc, kh, kw]
                    w_off = ic_offs * (OC * KHW) + oc * KHW + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)  # [BLOCK_IC]

                    # x[n, ic, ih, iw] -> [BLOCK_OHW, BLOCK_IC]
                    x_off = (n * IC * IHW
                             + ic_offs[None, :] * IHW
                             + ih[:, None] * IW
                             + iw[:, None])
                    x_mask = valid[:, None] & ic_mask[None, :]
                    x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                    tile_acc += tl.sum(x_vals * w_vals[None, :], axis=1)

        acc_scalar += tl.sum(tl.where(mask_ohw, tile_acc, 0.0), axis=0)

    s = tl.sum(acc_scalar, axis=0)
    bias = tl.load(bias_ptr + oc)
    total = (s + bias * OHW) * multiplier / OHW
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

        BLOCK_OHW = 256
        BLOCK_IC = 32

        grid = (N, OC)
        conv_transpose_mean_kernel_v2[grid](
            x, w, b, out,
            N, IC, OC, IH, IW, OH, OW,
            KH, KW,
            SH, SW, PH, PW,
            float(self.multiplier),
            BLOCK_OHW=BLOCK_OHW,
            BLOCK_IC=BLOCK_IC,
            num_warps=8,
            num_stages=2,
        )

        return out