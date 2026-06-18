import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W, OC, KH, KW, OH, OW,
    inv_npix,
    BLOCK_OC: tl.constexpr,
    BLOCK_PIX: tl.constexpr,
    IC_C: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    pix_offs = pid_p * BLOCK_PIX + tl.arange(0, BLOCK_PIX)

    oc_mask = oc_offs < OC
    pix_mask = pix_offs < (OH * OW)

    oh = pix_offs // OW
    ow = pix_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_PIX), dtype=tl.float32)

    # Iterate over IC, KH, KW (small, unrolled via constexpr)
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH_C):
            for kw in tl.static_range(0, KW_C):
                ih = oh + kh
                iw = ow + kw
                # x[n, ic, ih, iw]
                x_off = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                x_vals = tl.load(x_ptr + x_off, mask=pix_mask, other=0.0)  # [BLOCK_PIX]

                # w[oc, ic, kh, kw]
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_vals[:, None] * x_vals[None, :]

    # Add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_vals[:, None]

    # GELU (exact via erf)
    # gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    gelu_out = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Mask invalid pixels
    gelu_out = tl.where(pix_mask[None, :], gelu_out, 0.0)

    # Sum over pixels -> partial sum
    partial = tl.sum(gelu_out, axis=1)  # [BLOCK_OC]

    # Atomic add to output [N, OC], scaled by 1/npix
    partial = partial * inv_npix

    out_off = pid_n * OC + oc_offs
    tl.atomic_add(out_ptr + out_off, partial, mask=oc_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        npix = OH * OW
        inv_npix = 1.0 / npix

        out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 32
        BLOCK_PIX = 128

        grid = (
            N,
            (OC + BLOCK_OC - 1) // BLOCK_OC,
            (npix + BLOCK_PIX - 1) // BLOCK_PIX,
        )

        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, H, W, OC, KH, KW, OH, OW,
            inv_npix,
            BLOCK_OC=BLOCK_OC,
            BLOCK_PIX=BLOCK_PIX,
            IC_C=IC,
            KH_C=KH,
            KW_C=KW,
            num_warps=4,
            num_stages=2,
        )

        return out