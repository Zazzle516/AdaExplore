import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_scatter_kernel(
    x_ptr, w_ptr, partial_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW, SH, SW, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # One program per (n, ic, hw_tile, oc_tile)
    pid_n = tl.program_id(0)
    pid_ic = tl.program_id(1)
    pid_hw = tl.program_id(2)
    pid_oc = tl.program_id(3)

    hw_start = pid_hw * BLOCK_HW
    hw_offs = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < (IH * IW)
    ih = hw_offs // IW
    iw = hw_offs % IW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load input slice: x[n, ic, ih, iw] for each hw
    x_idx = ((pid_n * IC + pid_ic) * IH + ih) * IW + iw
    x_vals = tl.load(x_ptr + x_idx, mask=hw_mask, other=0.0)  # [BLOCK_HW]

    # Accumulator per oc - we'll accumulate scatter-sum into partial output
    # partial[n, ic_block? -> we use atomic add into (n, oc, 0)]
    # Instead, compute sum over output for this input-tile, per oc.
    # Each input element x[n,ic,ih,iw] scatters to outputs (n,oc, ih*SH-PH+kh, iw*SW-PW+kw)
    # with weight w[ic,oc,kh,kw]. The "mean" later sums all these. So we just need
    # sum over valid output positions of x * w. For each (ih,iw,kh,kw), check if
    # output position (ih*SH-PH+kh, iw*SW-PW+kw) is within [0,OH)x[0,OW).

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for kh in range(0, KH):
        oh = ih * SH - PH + kh
        h_valid = (oh >= 0) & (oh < OH)
        for kw in range(0, KW):
            ow = iw * SW - PW + kw
            w_valid = (ow >= 0) & (ow < OW)
            valid = hw_mask & h_valid & w_valid  # [BLOCK_HW]

            # weight w[ic, oc, kh, kw] -> shape [BLOCK_OC]
            w_idx = ((pid_ic * OC + oc_offs) * KH + kh) * KW + kw
            w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

            # Contribution: sum over hw of x_vals * valid, times w_vals
            x_masked = tl.where(valid, x_vals, 0.0)
            s = tl.sum(x_masked, axis=0)  # scalar
            acc += s * w_vals

    # Atomic add into partial[pid_n, oc_offs]
    out_idx = pid_n * OC + oc_offs
    tl.atomic_add(partial_ptr + out_idx, acc, mask=oc_mask)


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
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias  # [OC]

        # Partial sums of conv_transpose output along (H,W), shape [N, OC]
        partial = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 32
        BLOCK_HW = 128
        grid = (N, IC, triton.cdiv(IH * IW, BLOCK_HW), triton.cdiv(OC, BLOCK_OC))

        conv_transpose_scatter_kernel[grid](
            x, weight, partial,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
            BLOCK_OC=BLOCK_OC, BLOCK_HW=BLOCK_HW,
            num_warps=4,
        )

        # Add bias contribution: each bias element contributes bias[oc] * OH * OW per (n, oc)
        if bias is not None:
            partial = partial + bias.view(1, OC) * (OH * OW)

        # Multiply by multiplier and divide by (OH*OW) for the mean
        out = partial * (self.multiplier / (OH * OW))
        return out.view(N, OC, 1, 1)