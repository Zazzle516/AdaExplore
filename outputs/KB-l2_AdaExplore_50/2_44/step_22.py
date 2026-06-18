import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_scatter_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [IC, OC, KH, KW]
    partial_ptr,     # [N, OC, NUM_BLOCKS] - partial sums to be reduced
    N, IC, OC, IH, IW, OH, OW, KH, KW,
    STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    NUM_BLOCKS,
    BLOCK_OHW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # grid: (N, OC, NUM_BLOCKS)
    n = tl.program_id(0)
    oc = tl.program_id(1)
    blk = tl.program_id(2)

    ohw_start = blk * BLOCK_OHW
    offs = ohw_start + tl.arange(0, BLOCK_OHW)
    mask_ohw = offs < (OH * OW)
    oh = offs // OW
    ow = offs % OW

    acc = tl.zeros((BLOCK_OHW,), dtype=tl.float32)

    # For each output position, sum over IC, KH, KW
    # Standard conv_transpose2d: out[n,oc,oh,ow] = sum_{ic,kh,kw} x[n,ic,ih,iw] * w[ic,oc,kh,kw]
    # where ih = (oh + PAD_H - kh) / STRIDE_H, iw = (ow + PAD_W - kw) / STRIDE_W
    # valid if (oh + PAD_H - kh) % STRIDE_H == 0 and similarly for w
    for kh in range(0, KH):
        for kw in range(0, KW):
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

                # Load w[ic, oc, kh, kw] for ic in block -> [BLOCK_IC]
                w_off = ic_offs * (OC * KH * KW) + oc * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0)

                # Load x[n, ic, ih, iw] -> [BLOCK_OHW, BLOCK_IC]
                # x offset: n*IC*IH*IW + ic*IH*IW + ih*IW + iw
                x_off = (n * IC * IH * IW
                         + ic_offs[None, :] * (IH * IW)
                         + ih[:, None] * IW
                         + iw[:, None])
                x_mask = valid[:, None] & ic_mask[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # multiply and accumulate
                acc += tl.sum(x_vals * w_vals[None, :], axis=1)

    # Write partial sum for this block
    partial_off = n * (OC * NUM_BLOCKS) + oc * NUM_BLOCKS + blk
    block_sum = tl.sum(tl.where(mask_ohw, acc, 0.0), axis=0)
    tl.store(partial_ptr + partial_off, block_sum)


@triton.jit
def finalize_kernel(
    partial_ptr,   # [N, OC, NUM_BLOCKS]
    bias_ptr,      # [OC]
    out_ptr,       # [N, OC, 1, 1]
    N, OC, NUM_BLOCKS,
    OH, OW,
    multiplier,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    offs = tl.arange(0, BLOCK_R)
    mask = offs < NUM_BLOCKS
    base = n * (OC * NUM_BLOCKS) + oc * NUM_BLOCKS
    vals = tl.load(partial_ptr + base + offs, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)

    bias = tl.load(bias_ptr + oc)
    # sum over OH*OW of (conv + bias) * multiplier, then divide by (OH*OW)
    # mean = (s + bias * OH * OW) * multiplier / (OH*OW)
    total = (s + bias * (OH * OW)) * multiplier / (OH * OW)
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

        BLOCK_OHW = 128
        BLOCK_IC = 32
        NUM_BLOCKS = (OH * OW + BLOCK_OHW - 1) // BLOCK_OHW

        partial = torch.empty((N, OC, NUM_BLOCKS), device=x.device, dtype=torch.float32)

        grid = (N, OC, NUM_BLOCKS)
        conv_transpose_scatter_kernel[grid](
            x, w, partial,
            N, IC, OC, IH, IW, OH, OW, KH, KW,
            SH, SW, PH, PW,
            NUM_BLOCKS,
            BLOCK_OHW=BLOCK_OHW,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
        )

        out = torch.empty((N, OC, 1, 1), device=x.device, dtype=torch.float32)

        # next power of 2 >= NUM_BLOCKS
        BLOCK_R = 1
        while BLOCK_R < NUM_BLOCKS:
            BLOCK_R *= 2
        BLOCK_R = max(BLOCK_R, 16)

        finalize_kernel[(N * OC,)](
            partial, b, out,
            N, OC, NUM_BLOCKS,
            OH, OW,
            float(self.multiplier),
            BLOCK_R=BLOCK_R,
        )

        return out