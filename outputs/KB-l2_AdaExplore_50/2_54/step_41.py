import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 2, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    num_w_tiles = tl.cdiv(OW, BLOCK_OW)
    pid_oh = pid_sp // num_w_tiles
    pid_ow = pid_sp % num_w_tiles

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oh_offs = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)

    oc_mask = oc_offs < OC
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    IHW = IH * IW
    OHW = OH * OW
    KHW = KH * KW
    ICKHKW = IC * KHW

    acc = tl.zeros((BLOCK_OC, BLOCK_OH * BLOCK_OW), dtype=tl.float32)

    # spatial offsets within tile flatten
    oh_t = oh_offs[:, None]  # (BLOCK_OH, 1)
    ow_t = ow_offs[None, :]  # (1, BLOCK_OW)
    sp_mask_2d = oh_mask[:, None] & ow_mask[None, :]
    sp_mask = tl.reshape(sp_mask_2d, (BLOCK_OH * BLOCK_OW,))

    x_base = pid_n * IC * IHW

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_t + kh  # (BLOCK_OH, 1)
                iw = ow_t + kw  # (1, BLOCK_OW)
                x_idx = x_base + ic * IHW + ih * IW + iw  # (BLOCK_OH, BLOCK_OW)
                x_vals_2d = tl.load(x_ptr + x_idx, mask=sp_mask_2d, other=0.0)
                x_vals = tl.reshape(x_vals_2d, (BLOCK_OH * BLOCK_OW,))

                w_idx = oc_offs * ICKHKW + ic * KHW + kh * KW + kw
                w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                acc += w_vals[:, None] * x_vals[None, :]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    mult = tl.load(mult_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc * mult[:, None]

    acc = tl.where(acc >= 0, acc, acc * 0.01)

    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    sp_idx_2d = oh_t * OW + ow_t  # (BLOCK_OH, BLOCK_OW)
    sp_idx = tl.reshape(sp_idx_2d, (BLOCK_OH * BLOCK_OW,))
    out_idx = pid_n * OC * OHW + oc_offs[:, None] * OHW + sp_idx[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        mult = self.multiplier.contiguous().view(-1)

        N, IC, IH, IW = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        def grid(meta):
            return (
                N,
                triton.cdiv(OC, meta['BLOCK_OC']),
                triton.cdiv(OH, meta['BLOCK_OH']) * triton.cdiv(OW, meta['BLOCK_OW']),
            )

        conv_fused_kernel[grid](
            x, w, b, mult, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH=KH, KW=KW,
        )
        return out