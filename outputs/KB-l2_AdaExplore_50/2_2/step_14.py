import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose2d_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    inv_scale: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program_id 0: (oc tile, sp tile)
    # program_id 1: n
    pid = tl.program_id(0)
    n = tl.program_id(1)

    num_sp_tiles = tl.cdiv(OH * OW, BLOCK_SP)
    oc_tile = pid // num_sp_tiles
    sp_tile = pid % num_sp_tiles

    offs_oc = oc_tile * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_sp = sp_tile * BLOCK_SP + tl.arange(0, BLOCK_SP)

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < (OH * OW)

    oh = offs_sp // OW
    ow = offs_sp % OW

    # Padded input position: ph = oh + PAD_H, pw = ow + PAD_W
    ph = oh + PAD_H
    pw = ow + PAD_W

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # iterate over kh, kw
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # input position: ih*STRIDE_H = ph - kh => ih = (ph - kh)/STRIDE_H
            ih_num = ph - kh
            iw_num = pw - kw
            ih = ih_num // STRIDE_H
            iw = iw_num // STRIDE_W
            valid = ((ih_num % STRIDE_H) == 0) & ((iw_num % STRIDE_W) == 0) \
                    & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            valid = valid & mask_sp

            # Loop over IC
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)
                mask_ic = offs_ic < IC

                # Load x[n, ic, ih, iw] -> shape (BLOCK_SP, BLOCK_IC)
                x_offsets = (n * IC * IH * IW
                             + offs_ic[None, :] * IH * IW
                             + ih[:, None] * IW
                             + iw[:, None])
                x_mask = valid[:, None] & mask_ic[None, :]
                x_vals = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)

                # Load w[ic, oc, kh, kw] -> shape (BLOCK_IC, BLOCK_OC)
                w_offsets = (offs_ic[:, None] * OC * KH * KW
                             + offs_oc[None, :] * KH * KW
                             + kh * KW + kw)
                w_mask = mask_ic[:, None] & mask_oc[None, :]
                w_vals = tl.load(w_ptr + w_offsets, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Add conv bias (b_ptr) per oc and extra bias_ptr per oc
    cb = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    eb = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + cb[None, :] + eb[None, :]

    # clamp(0,1) -> *scale -> clamp(0,1) -> /scale
    # Equivalent: clamp(acc, 0, 1) then min(scaled,1)/scale
    # Since scaling_factor>=1 typically, after *scale clamp to 1, divide by scale gives min(clamped, 1/scale)
    # But to be exact: do the full chain
    y = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    y = y * (1.0 / inv_scale)  # placeholder, we'll fix below
    # Actually use scaling_factor passed: inv_scale here is scaling_factor
    # rewrite properly:
    y = tl.minimum(tl.maximum(acc, 0.0), 1.0) * inv_scale
    y = tl.minimum(tl.maximum(y, 0.0), 1.0) * (1.0 / inv_scale)

    # Store to out[n, oc, oh, ow]
    out_offsets = (n * OC * OH * OW
                   + offs_oc[None, :] * OH * OW
                   + offs_sp[:, None])
    out_mask = mask_sp[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offsets, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW
        OC = self.out_channels

        weight = self.conv_transpose.weight.contiguous()  # (IC, OC, KH, KW)
        conv_bias = self.conv_transpose.bias.contiguous()  # (OC,)
        extra_bias = self.bias.view(-1).contiguous()  # (OC,)

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64
        BLOCK_IC = 32

        num_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC
        num_sp_tiles = (OH * OW + BLOCK_SP - 1) // BLOCK_SP

        grid = (num_oc_tiles * num_sp_tiles, N)

        conv_transpose2d_kernel[grid](
            x, weight, conv_bias, extra_bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            float(self.scaling_factor),
            BLOCK_OC, BLOCK_SP, BLOCK_IC,
            num_warps=4, num_stages=2,
        )
        return out