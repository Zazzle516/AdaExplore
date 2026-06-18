import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy: Fused ConvTranspose2d + multiply + global mean.
# 
# Key observation: the final output mean[n,oc] = (1/HW) * sum_{ih,iw,ic,kh,kw}
#   x[n,ic,ih,iw] * w[ic,oc,kh,kw] * valid(ih,iw,kh,kw) + bias[oc]
# 
# We must visit every (ic,ih,iw,kh,kw) tuple to preserve MAC count.
# We use tl.dot to do the IC reduction efficiently.
#
# For each (kh, kw), validity depends on (ih, iw). For interior input positions
# (those for which all kh,kw map to valid oh,ow), we can avoid the per-(kh,kw)
# masking entirely. We split: for each (kh, kw) loop body, compute valid mask
# and do a matmul over IC.
#
# Better: combine all (kh,kw) by treating the weight as [IC, OC*KH*KW] and
# reduce. But we still need to mask x per (kh,kw) because validity differs.
# 
# We instead pre-mask x per (kh,kw): for each (kh,kw), the valid input region is
# the same shape as input but with boundary masking. We use tl.dot for IC reduction.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256, 'BLOCK_IC': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_IC': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_IC': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_IC': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_IC': 32, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_IC': 64, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_IC': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256, 'BLOCK_IC': 64, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 512, 'BLOCK_IC': 16, 'BLOCK_OC': 128}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'IH', 'IW'],
)
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
    ic_range = tl.arange(0, BLOCK_IC)

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    HW = IH * IW
    KHW: tl.constexpr = KH * KW

    # Loop over IC blocks (outer) so weights are loaded once per ic block
    for ic_start in range(0, IC, BLOCK_IC):
        ic_offs = ic_start + ic_range
        ic_mask = ic_offs < IC

        # Load all KH*KW weight tiles into a single [BLOCK_IC, BLOCK_OC*KHW] tensor
        # Layout: w[ic, oc, kh, kw]; we load [BLOCK_IC, BLOCK_OC, KH, KW]
        # Flatten to [BLOCK_IC, BLOCK_OC * KHW]
        khw_range = tl.arange(0, KHW)
        w_off = (ic_offs[:, None, None] * (OC * KHW)
                 + oc_offs[None, :, None] * KHW
                 + khw_range[None, None, :])
        w_m = ic_mask[:, None, None] & oc_mask[None, :, None]
        w_all = tl.load(w_ptr + w_off, mask=w_m, other=0.0)  # [BLOCK_IC, BLOCK_OC, KHW]

        # Loop over HW tiles
        for hw_start in range(0, HW, BLOCK_HW):
            idx = hw_start + hw_offs
            hw_mask = idx < HW
            ih = idx // IW
            iw = idx % IW

            x_off = (pid_n * IC * HW
                     + ic_offs[:, None] * HW
                     + idx[None, :])
            x_m = ic_mask[:, None] & hw_mask[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)  # [BLOCK_IC, BLOCK_HW]

            for kh in tl.static_range(0, KH):
                oh = ih * STRIDE - PAD + kh
                oh_valid = (oh >= 0) & (oh < OH)
                for kw in tl.static_range(0, KW):
                    ow = iw * STRIDE - PAD + kw
                    ow_valid = (ow >= 0) & (ow < OW)
                    valid = hw_mask & oh_valid & ow_valid

                    x_masked = tl.where(valid[None, :], x_tile, 0.0)
                    x_sum = tl.sum(x_masked, axis=1)  # [BLOCK_IC]

                    khw_idx = kh * KW + kw
                    # Slice w_all[:, :, khw_idx] -> [BLOCK_IC, BLOCK_OC]
                    # Use mask trick: select via tl.sum over a one-hot
                    one_hot = (khw_range == khw_idx).to(tl.float32)
                    w_tile = tl.sum(w_all * one_hot[None, None, :], axis=2)

                    contrib = tl.sum(w_tile * x_sum[:, None], axis=0)
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

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_OC']))

    fused_kernel[grid](
        x, weight, bias, out,
        N, IC, IH, IW, OC, OH, OW,
        KH, KW,
        stride, padding,
        float(multiplier),
        1.0 / (OH * OW),
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