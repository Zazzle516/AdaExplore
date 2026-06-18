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
    inv_ohw = 1.0 / OHW

    acc = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

    # Loop over output spatial in tiles
    for ohw_start in range(0, OHW, BLOCK_OHW):
        offs = ohw_start + tl.arange(0, BLOCK_OHW)
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
                valid = ((num_h % STRIDE_H) == 0) & ((num_w % STRIDE_W) == 0) \
                        & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & mask_ohw

                # Loop over IC
                for ic in range(0, IC):
                    w_off = ic * (OC * KH * KW) + oc * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off)

                    x_off = (n * IC * IH * IW
                             + ic * (IH * IW)
                             + ih * IW
                             + iw)
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)

                    tile_acc += x_val * w_val

        acc += tl.where(mask_ohw, tile_acc, 0.0)

    s = tl.sum(acc, axis=0)
    bias = tl.load(bias_ptr + oc)
    total = (s + bias * OHW) * multiplier * inv_ohw
    tl.store(out_ptr + n * OC + oc, total)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OHW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OHW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OHW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OHW': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OHW': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OHW': 1024}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OH', 'OW'],
)
@triton.jit
def conv_transpose_mean_kernel_v3(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    out_ptr,         # [N, OC]  (accumulator; bias added in post step)
    N, IC, OC, IH, IW, OH, OW,
    NUM_TILES,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_IC: tl.constexpr,
    BLOCK_OHW: tl.constexpr,
):
    # grid: (N * OC, NUM_TILES)
    pid_noc = tl.program_id(0)
    tile_id = tl.program_id(1)
    n = pid_noc // OC
    oc = pid_noc % OC

    OHW = OH * OW

    ohw_start = tile_id * BLOCK_OHW
    offs = ohw_start + tl.arange(0, BLOCK_OHW)
    mask_ohw = offs < OHW
    oh = offs // OW
    ow = offs % OW

    tile_acc = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

    ic_offs = tl.arange(0, BLOCK_IC)
    ic_mask = ic_offs < IC

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            num_h = oh + PAD_H - kh
            num_w = ow + PAD_W - kw
            ih = num_h // STRIDE_H
            iw = num_w // STRIDE_W
            valid = ((num_h % STRIDE_H) == 0) & ((num_w % STRIDE_W) == 0) \
                    & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & mask_ohw

            # weight[:, oc, kh, kw] for all ic in [0, BLOCK_IC)
            w_off = ic_offs * (OC * KH * KW) + oc * (KH * KW) + kh * KW + kw
            w_vals = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)

            x_off = (n * IC * IH * IW
                     + ic_offs[None, :] * (IH * IW)
                     + ih[:, None] * IW
                     + iw[:, None])
            x_mask = valid[:, None] & ic_mask[None, :]
            x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

            tile_acc += tl.sum(x_vals * w_vals[None, :], axis=1)

    s = tl.sum(tl.where(mask_ohw, tile_acc, 0.0), axis=0)
    tl.atomic_add(out_ptr + n * OC + oc, s)


@triton.jit
def finalize_kernel(
    acc_ptr,     # [N, OC]
    bias_ptr,    # [OC]
    out_ptr,     # [N, OC, 1, 1]
    N, OC,
    inv_ohw, multiplier,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N * OC
    oc = offs % OC
    s = tl.load(acc_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + oc, mask=mask, other=0.0)
    out = (s * inv_ohw + b) * multiplier
    tl.store(out_ptr + offs, out, mask=mask)


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

        # BLOCK_IC must be a power of 2 >= IC for the single-tile load.
        BLOCK_IC = 1
        while BLOCK_IC < IC:
            BLOCK_IC *= 2

        OHW = OH * OW

        acc = torch.zeros((N, OC), device=x.device, dtype=torch.float32)
        out = torch.empty((N, OC, 1, 1), device=x.device, dtype=torch.float32)

        def grid(meta):
            return (N * OC, triton.cdiv(OHW, meta['BLOCK_OHW']))

        conv_transpose_mean_kernel_v3[grid](
            x, w, acc,
            N, IC, OC, IH, IW, OH, OW,
            0,  # NUM_TILES (unused inside kernel; grid computed externally)
            KH, KW,
            SH, SW, PH, PW,
            BLOCK_IC,
        )

        # finalize: add bias, multiply
        total = N * OC
        BLOCK = 128
        finalize_kernel[(triton.cdiv(total, BLOCK),)](
            acc, b, out,
            N, OC,
            1.0 / float(OHW), float(self.multiplier),
            BLOCK=BLOCK,
        )

        return out