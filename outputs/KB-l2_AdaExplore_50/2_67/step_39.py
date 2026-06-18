import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PIX': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PIX': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PIX': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PIX': 512}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC_C'],
)
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

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    npix = OH * OW

    # Preload bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Accumulator over pixels: per-OC scalar sum of gelu outputs
    sum_acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    inv_sqrt2 = 0.70710678118654752440

    n_pix_tiles = tl.cdiv(npix, BLOCK_PIX)
    for tile in range(0, n_pix_tiles):
        pix_offs = tile * BLOCK_PIX + tl.arange(0, BLOCK_PIX)
        pix_mask = pix_offs < npix

        oh = pix_offs // OW
        ow = pix_offs % OW

        acc = tl.zeros((BLOCK_OC, BLOCK_PIX), dtype=tl.float32)

        for ic in tl.static_range(0, IC_C):
            for kh in tl.static_range(0, KH_C):
                for kw in tl.static_range(0, KW_C):
                    ih = oh + kh
                    iw = ow + kw
                    x_off = pid_n * (IC * H * W) + ic * (H * W) + ih * W + iw
                    x_vals = tl.load(x_ptr + x_off, mask=pix_mask, other=0.0)
                    w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                    w_col = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += w_col[:, None] * x_vals[None, :]

        acc = acc + b_vals[:, None]
        gelu_out = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
        gelu_out = tl.where(pix_mask[None, :], gelu_out, 0.0)
        sum_acc += tl.sum(gelu_out, axis=1)

    out_vals = sum_acc * inv_npix
    out_off = pid_n * OC + oc_offs
    tl.store(out_ptr + out_off, out_vals, mask=oc_mask)


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

        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        grid = lambda META: (
            N,
            (OC + META['BLOCK_OC'] - 1) // META['BLOCK_OC'],
        )

        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, H, W, OC, KH, KW, OH, OW,
            inv_npix,
            IC_C=IC,
            KH_C=KH,
            KW_C=KW,
        )

        return out