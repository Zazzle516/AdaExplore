import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused conv-transpose + multiplier + spatial mean.
# One program per (N, OC_tile). It loops over all output (oh, ow) positions,
# accumulates the conv-transpose result into a register vector of size BLOCK_OC,
# and writes only the per-(n, oc) sum. We never materialize the (N, OC, OH, OW)
# intermediate.
#
# For each output position (oh, ow), the conv-transpose computes:
#   y[n, oc, oh, ow] = sum_{ic, kh, kw valid} x[n, ic, ih, iw] * w[ic, oc, kh, kw]
# where ih = (oh + PH - kh) / SH (if divisible) and iw similar.
#
# Sum over (oh, ow):
#   S[n, oc] = sum_{oh, ow} y[n, oc, oh, ow]
#            = sum_{ic, kh, kw} w[ic, oc, kh, kw] *
#                sum_{(oh, ow) valid for (kh,kw)} x[n, ic, ih(oh), iw(ow)]
#
# So for fixed (kh, kw), as (oh, ow) varies, (ih, iw) sweeps a contiguous
# rectangle of the input. The inner sum is sum over a subrectangle of
# x[n, ic, :, :].
#
# IMPORTANT — Safety contract: per the rules we cannot precompute this
# subrectangle sum at init time, nor fold it into the weight, nor reduce x along
# any axis before the GEMM. We MUST perform asymptotic IC*OC*OH*OW*KH*KW work.
# So we do the full conv-transpose multiply-adds inside the kernel — one program
# accumulates contributions over all output positions for its (N, OC tile).


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64, 'IC_TILE': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'IC_TILE': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'IC_TILE': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128, 'IC_TILE': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'IC_TILE': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64, 'IC_TILE': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'IC_TILE': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 256, 'IC_TILE': 16}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def fused_convt_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    multiplier, inv_area,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    IC_TILE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Final per-(oc) accumulator (sum over all oh*ow and ic, kh, kw).
    sum_acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    HW = OH * OW
    hw_offs_block = tl.arange(0, BLOCK_HW)
    ic_offs = tl.arange(0, IC_TILE)

    # Loop over output spatial tiles.
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + hw_offs_block
        hw_mask = hw_offs < HW

        oh = hw_offs // OW
        ow = hw_offs % OW

        acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih_num = oh + PAD_H - kh
                iw_num = ow + PAD_W - kw
                ih_num_c = tl.where(ih_num >= 0, ih_num, 0)
                iw_num_c = tl.where(iw_num >= 0, iw_num, 0)
                ih = ih_num_c // STRIDE_H
                iw = iw_num_c // STRIDE_W
                valid = (ih_num >= 0) & (iw_num >= 0) & \
                        ((ih_num % STRIDE_H) == 0) & ((iw_num % STRIDE_W) == 0) & \
                        (ih < IH) & (iw < IW) & hw_mask

                for ic_start in range(0, IC, IC_TILE):
                    ic_idx = ic_start + ic_offs
                    ic_mask = ic_idx < IC

                    # x[n, ic_tile, ih, iw] -> [IC_TILE, BLOCK_HW]
                    x_idx = pid_n * (IC * IH * IW) + ic_idx[:, None] * (IH * IW) + (ih * IW + iw)[None, :]
                    x_mask = ic_mask[:, None] & valid[None, :]
                    x_tile = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

                    # w[ic_tile, oc, kh, kw] -> [IC_TILE, BLOCK_OC]
                    w_idx = ic_idx[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + (kh * KW + kw)
                    w_mask = ic_mask[:, None] & oc_mask[None, :]
                    w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

                    acc += tl.dot(tl.trans(w_tile), x_tile)

        # acc is [BLOCK_OC, BLOCK_HW]; add bias (only the valid hw positions
        # should contribute to the mean, so mask invalid columns).
        bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc = acc + bias[:, None]
        # mask out invalid hw positions
        acc = tl.where(hw_mask[None, :], acc, 0.0)
        # accumulate sum over hw for each oc
        sum_acc += tl.sum(acc, axis=1)

    # Now sum_acc[oc] = sum over (oh, ow) of (conv_transpose output + bias).
    # Final mean = sum_acc * multiplier / (OH * OW)
    result = sum_acc * (multiplier * inv_area)

    out_idx = pid_n * OC + oc_offs
    tl.store(out_ptr + out_idx, result, mask=oc_mask)


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
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        out = torch.empty((N, OC), device=x.device, dtype=x.dtype)

        grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']))

        fused_convt_mean_kernel[grid](
            x, weight, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            float(self.multiplier), 1.0 / float(OH * OW),
        )

        return out.view(N, OC, 1, 1)