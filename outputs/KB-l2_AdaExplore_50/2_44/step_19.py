import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose2d + multiplier + spatial mean.
# One program per (N, OC tile). Iterate over input spatial positions (ih, iw)
# and kernel taps (kh, kw); accumulate contributions directly into per-(n, oc) scalar.
#
# For each (n, ic, ih, iw, kh, kw):
#   oh = ih * SH - PH + kh, ow = iw * SW - PW + kw
#   contribution to out[n, oc] += x[n,ic,ih,iw] * w[ic,oc,kh,kw]  if 0<=oh<OH and 0<=ow<OW
# Then divide by (OH*OW) and multiply by multiplier.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'IH', 'IW', 'KH', 'KW'],
)
@triton.jit
def fused_convtrans_mean_kernel(
    x_ptr,           # [N, IH, IW, IC]  (channels-last)
    w_ptr,           # [IC, OC, KH, KW] (original layout from nn.ConvTranspose2d)
    b_ptr,           # [OC]
    out_ptr,         # [N, OC]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    inv_area, multiplier,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)  # per-oc accumulator

    ic_arange = tl.arange(0, BLOCK_IC)

    # Loop over input spatial positions
    for ih in range(0, IH):
        for iw in range(0, IW):
            # Loop over input channel tiles
            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + ic_arange
                ic_mask = ic_idx < IC

                # x[n, ih, iw, ic_tile]
                x_offs = pid_n * (IH * IW * IC) + ih * (IW * IC) + iw * IC + ic_idx
                x_vec = tl.load(x_ptr + x_offs, mask=ic_mask, other=0.0)  # [BLOCK_IC]

                # Sum over kernel taps for valid (oh, ow)
                # For each (kh, kw), w[ic_tile, oc_tile, kh, kw] -> [BLOCK_IC, BLOCK_OC]
                # Weighted by x_vec[:, None], then mask by validity (a scalar per kh,kw).
                for kh in tl.static_range(0, KH):
                    oh = ih * SH - PH + kh
                    valid_h = (oh >= 0) & (oh < OH)
                    for kw in tl.static_range(0, KW):
                        ow = iw * SW - PW + kw
                        valid = valid_h & (ow >= 0) & (ow < OW)
                        # Load weights w[ic_tile, oc_tile, kh, kw]
                        w_offs = ic_idx[:, None] * (OC * KH * KW) + oc_offs[None, :] * (KH * KW) + (kh * KW + kw)
                        w_mask = ic_mask[:, None] & oc_mask[None, :]
                        w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_IC, BLOCK_OC]

                        # Contribution: sum over ic of x_vec[ic] * w_tile[ic, oc]
                        contrib = tl.sum(x_vec[:, None] * w_tile, axis=0)  # [BLOCK_OC]
                        contrib = tl.where(valid, contrib, 0.0)
                        acc += contrib

    # Bias is added to every spatial position, so mean adds bias as-is.
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    mean_val = acc * inv_area + bias
    mean_val = mean_val * multiplier

    out_offs = pid_n * OC + oc_offs
    tl.store(out_ptr + out_offs, mean_val, mask=oc_mask)


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
        x = x.cuda()
        weight = self.conv_transpose.weight.contiguous().cuda()  # [IC, OC, KH, KW]
        bias = self.conv_transpose.bias.contiguous().cuda()      # [OC]

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        # Convert input to channels-last layout: [N, IH, IW, IC]
        x_cl = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        inv_area = 1.0 / float(OH * OW)

        grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']))

        fused_convtrans_mean_kernel[grid](
            x_cl, weight, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            inv_area, float(self.multiplier),
        )

        return out.view(N, OC, 1, 1)