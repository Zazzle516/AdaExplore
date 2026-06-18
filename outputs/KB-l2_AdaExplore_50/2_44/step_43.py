import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Key insight: For ConvTranspose2d followed by global mean pooling, the output
# is sum over (oh, ow) of out[n, oc, oh, ow]. Since each input position
# (n, ic, ih, iw) with weight w[ic, oc, kh, kw] contributes to output
# (oh, ow) = (ih*S - P + kh, iw*S - P + kw), the contribution to the sum is:
#   x[n,ic,ih,iw] * w[ic,oc,kh,kw] * valid_mask(ih,iw,kh,kw)
# where valid_mask depends only on (ih, iw, kh, kw), NOT on (n, oc).
#
# So mean = (1/HW) * sum_{ic,ih,iw,kh,kw} x[n,ic,ih,iw] * w[ic,oc,kh,kw] * valid
#         = (1/HW) * sum_{ic} sum_{ih,iw} x[n,ic,ih,iw] * sum_{kh,kw} w[ic,oc,kh,kw] * valid(ih,iw,kh,kw)
#
# However, per the safety contract, we cannot pre-reduce w over (kh,kw).
# We must visit every (ic, ih, iw, kh, kw) tuple to preserve MAC count.
#
# The fastest approach: parallelize over (N, OC tile) AND tile over input HW,
# and use tl.dot for the matmul over IC dimension. We need a 2D parallelism
# over OC and HW tiles, then reduce over IC and kernel.

@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW, OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    multiplier,
    inv_HW,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    hw_offs = tl.arange(0, BLOCK_HW)

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    HW = IH * IW

    # Loop over input spatial tiles
    for hw_start in range(0, HW, BLOCK_HW):
        idx = hw_start + hw_offs
        hw_mask = idx < HW
        ih = idx // IW
        iw = idx % IW

        # Compute per-(ih,iw,kh,kw) validity once
        # We'll accumulate sum over kh,kw of valid weighted contribution per (oc, hw_tile)
        # Strategy: for each (kh, kw), gather x and weight, do outer product, mask, accumulate

        # Loop over IC in blocks
        for ic_start in range(0, IC, BLOCK_IC):
            ic_offs = ic_start + tl.arange(0, BLOCK_IC)
            ic_mask = ic_offs < IC

            # Load x[n, ic_block, ih, iw] -> [BLOCK_IC, BLOCK_HW]
            x_off = (pid_n * IC * HW
                     + ic_offs[:, None] * HW
                     + idx[None, :])
            x_m = ic_mask[:, None] & hw_mask[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)  # [BLOCK_IC, BLOCK_HW]

            # For each (kh, kw), compute weight and validity
            for kh in tl.static_range(0, KH):
                oh = ih * STRIDE - PAD + kh
                oh_valid = (oh >= 0) & (oh < OH)
                for kw in tl.static_range(0, KW):
                    ow = iw * STRIDE - PAD + kw
                    ow_valid = (ow >= 0) & (ow < OW)
                    valid = hw_mask & oh_valid & ow_valid  # [BLOCK_HW]

                    # Load w[ic_block, oc_block, kh, kw] -> [BLOCK_IC, BLOCK_OC]
                    w_off = (ic_offs[:, None] * (OC * KH * KW)
                             + oc_offs[None, :] * (KH * KW)
                             + kh * KW + kw)
                    w_m = ic_mask[:, None] & oc_mask[None, :]
                    w_tile = tl.load(w_ptr + w_off, mask=w_m, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                    # Masked x: [BLOCK_IC, BLOCK_HW]
                    x_masked = tl.where(valid[None, :], x_tile, 0.0)

                    # Sum over HW per IC: [BLOCK_IC]
                    x_sum_per_ic = tl.sum(x_masked, axis=1)  # [BLOCK_IC]

                    # contribution to acc[oc] = sum_ic w[ic,oc] * x_sum_per_ic[ic]
                    contrib = tl.sum(w_tile * x_sum_per_ic[:, None], axis=0)  # [BLOCK_OC]
                    acc += contrib

    b = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    mean_val = (acc * inv_HW + b) * multiplier

    out_off = pid_n * OC + oc_offs
    tl.store(out_ptr + out_off, mean_val, mask=oc_mask)


def fused_convtranspose_mean(x, weight, bias, stride, padding, output_padding, multiplier):
    N, IC, IH, IW = x.shape
    IC_w, OC, KH, KW = weight.shape
    OH = (IH - 1) * stride - 2 * padding + KH + output_padding
    OW = (IW - 1) * stride - 2 * padding + KW + output_padding

    out = torch.empty((N, OC, 1, 1), device=x.device, dtype=x.dtype)

    BLOCK_OC = 32
    BLOCK_HW = 256
    BLOCK_IC = 16

    grid = (N, triton.cdiv(OC, BLOCK_OC))

    fused_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW, OC, OH, OW,
        KH, KW,
        stride, padding,
        float(multiplier),
        1.0 / (OH * OW),
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=BLOCK_HW,
        BLOCK_IC=BLOCK_IC,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, multiplier):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.multiplier = multiplier
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()
        bias = self.conv_transpose.bias.contiguous().cuda()
        return fused_convtranspose_mean(
            x, weight, bias,
            self.stride, self.padding, self.output_padding,
            self.multiplier,
        )