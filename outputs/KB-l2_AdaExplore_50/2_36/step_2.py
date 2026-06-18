import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_transpose_min_sum_gelu_kernel(
    x_ptr, w_ptr, b_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, OW_tiles)
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_start = pid_w * BLOCK_W
    ow_offs = w_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, BLOCK_OC)  # assume BLOCK_OC >= OC
    oc_mask = oc_offs < OC

    # Per-OW running sum_oh of min_oc(conv[n,oc,oh,ow])
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    # Load conv bias (OC,) -> broadcast later
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # (BLOCK_OC,)

    # iw candidates per kw: iw_num = ow + PW - kw, must be divisible by SW
    # ih candidates per kh: ih_num = oh + PH - kh, must be divisible by SH

    for oh in range(0, OH):
        # Initialize conv tile [BLOCK_OC, BLOCK_W] with bias broadcast
        conv_tile = b_vals[:, None] + tl.zeros((BLOCK_OC, BLOCK_W), dtype=tl.float32)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            ih_ok = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih >= 0) & (ih < IH)

            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PW - kw
                iw = iw_num // SW
                iw_ok = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw >= 0) & (iw < IW)
                col_valid = iw_ok & ow_mask  # (BLOCK_W,)

                if ih_ok:
                    # Loop over IC, accumulating into conv_tile
                    # Load x[n, ic, ih, iw] for ic in [0,IC), iw broadcast over BLOCK_W
                    # Then w[ic, oc, kh, kw] for ic in [0,IC), oc in [0, BLOCK_OC)
                    # Use matmul-style: conv_tile += w_T (BLOCK_OC, IC) @ x (IC, BLOCK_W)
                    for ic in range(0, IC):
                        x_off = (pid_n * IC * IH * IW
                                 + ic * IH * IW
                                 + ih * IW
                                 + iw)
                        x_vals = tl.load(x_ptr + x_off, mask=col_valid, other=0.0)  # (BLOCK_W,)

                        w_off = (ic * OC * KH * KW
                                 + oc_offs * KH * KW
                                 + kh * KW
                                 + kw)
                        w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # (BLOCK_OC,)

                        conv_tile += w_vals[:, None] * x_vals[None, :]

        # Mask out invalid OC rows with +inf so they don't affect min
        conv_tile = tl.where(oc_mask[:, None], conv_tile, float('inf'))
        # min over OC -> (BLOCK_W,)
        min_vec = tl.min(conv_tile, axis=0)
        acc += min_vec

    # GELU
    inv_sqrt2 = 0.7071067811865475
    gelu_out = 0.5 * acc * (1.0 + tl.math.erf(acc * inv_sqrt2))

    bias_val = tl.load(bias_ptr)
    result = gelu_out + bias_val

    out_offs = pid_n * OW + ow_offs
    tl.store(out_ptr + out_offs, result, mask=ow_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OP = self.output_padding
        OH = (IH - 1) * SH - 2 * PH + KH + OP
        OW = (IW - 1) * SW - 2 * PW + KW + OP

        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        bias = self.bias.contiguous()

        out = torch.empty((N, 1, 1, OW), device=x.device, dtype=x.dtype)

        # pick BLOCK_OC as next pow2 >= OC
        def next_pow2(n):
            p = 1
            while p < n:
                p *= 2
            return p

        BLOCK_OC = next_pow2(OC)
        BLOCK_W = 64

        grid = (N, triton.cdiv(OW, BLOCK_W))

        conv_transpose_min_sum_gelu_kernel[grid](
            x, weight, conv_bias, bias, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW, SH, SW, PH, PW,
            BLOCK_W=BLOCK_W,
            BLOCK_OC=BLOCK_OC,
            num_warps=8,
            num_stages=2,
        )

        return out