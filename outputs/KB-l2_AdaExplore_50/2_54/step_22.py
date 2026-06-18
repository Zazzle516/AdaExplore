import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_fused_kernel(
    x_ptr,        # NHWC input [N, H, W, IC]
    w_ptr,        # weights [OC, KH, KW, IC]
    bias_ptr,     # [OC] fused bias = conv_bias * multiplier
    scale_ptr,    # [OC] = multiplier
    out_ptr,      # NCHW output [N, OC, OH, OW]
    N, H, W, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)             # batch
    pid_sp = tl.program_id(1)            # spatial tile
    pid_oc = tl.program_id(2)            # OC tile

    n_tiles_w = tl.cdiv(OW, BLOCK_OW)
    tile_oh = pid_sp // n_tiles_w
    tile_ow = pid_sp % n_tiles_w

    oh_start = tile_oh * BLOCK_OH
    ow_start = tile_ow * BLOCK_OW
    oc_start = pid_oc * BLOCK_OC

    offs_oh = oh_start + tl.arange(0, BLOCK_OH)   # [BLOCK_OH]
    offs_ow = ow_start + tl.arange(0, BLOCK_OW)   # [BLOCK_OW]
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)   # [BLOCK_OC]

    mask_oh = offs_oh < OH
    mask_ow = offs_ow < OW
    mask_oc = offs_oc < OC

    # Flatten spatial tile to BLOCK_M = BLOCK_OH * BLOCK_OW
    # We treat the spatial tile as a 1D set of rows.
    # row m -> (oh_m, ow_m)
    oh_m = (tl.arange(0, BLOCK_OH * BLOCK_OW) // BLOCK_OW)  # [M]
    ow_m = (tl.arange(0, BLOCK_OH * BLOCK_OW) % BLOCK_OW)   # [M]
    oh_idx = oh_start + oh_m   # [M]
    ow_idx = ow_start + ow_m   # [M]
    m_mask = (oh_idx < OH) & (ow_idx < OW)

    acc = tl.zeros((BLOCK_OH * BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    offs_ic = tl.arange(0, BLOCK_IC)  # [BLOCK_IC]

    # Input base for this batch
    x_base = pid_n * (H * W * IC)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh_idx + kh   # [M]
            iw = ow_idx + kw   # [M]
            # ih, iw always valid for valid output (no padding)
            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + offs_ic       # [BLOCK_IC]
                ic_mask = ic_idx < IC

                # x[n, ih, iw, ic] -> [M, BLOCK_IC]
                x_offs = (x_base
                          + ih[:, None] * (W * IC)
                          + iw[:, None] * IC
                          + ic_idx[None, :])
                x_m = m_mask[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_offs, mask=x_m, other=0.0)

                # w[oc, kh, kw, ic] -> [BLOCK_IC, BLOCK_OC]
                # weight layout: [OC, KH, KW, IC], stride (KH*KW*IC, KW*IC, IC, 1)
                w_offs = (offs_oc[None, :] * (KH * KW * IC)
                          + kh * (KW * IC)
                          + kw * IC
                          + ic_idx[:, None])
                w_m = ic_mask[:, None] & mask_oc[None, :]
                w_tile = tl.load(w_ptr + w_offs, mask=w_m, other=0.0)

                acc += tl.dot(x_tile, w_tile)

    # Epilogue
    scale = tl.load(scale_ptr + offs_oc, mask=mask_oc, other=0.0)   # [BLOCK_OC]
    bias = tl.load(bias_ptr + offs_oc, mask=mask_oc, other=0.0)     # [BLOCK_OC]

    y = acc * scale[None, :] + bias[None, :]
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.7071067811865475
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    # Store to NCHW: out[n, oc, oh, ow]
    # offsets: pid_n*OC*OH*OW + oc*OH*OW + oh*OW + ow
    out_offs = (pid_n * (OC * OH * OW)
                + offs_oc[None, :] * (OH * OW)
                + oh_idx[:, None] * OW
                + ow_idx[:, None])
    store_mask = m_mask[:, None] & mask_oc[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        N, C, H, W_ = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W_ - KW + 1

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w = self.conv.weight  # [OC, IC, KH, KW]
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # [OC, KH, KW, IC]

        mult_flat = self.multiplier.contiguous().view(-1)  # [OC]
        conv_bias = self.conv.bias  # [OC]
        fused_bias = conv_bias * mult_flat

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        # Tile config
        BLOCK_OH = 8
        BLOCK_OW = 32
        BLOCK_OC = 64
        BLOCK_IC = 32

        n_tiles_h = (OH + BLOCK_OH - 1) // BLOCK_OH
        n_tiles_w = (OW + BLOCK_OW - 1) // BLOCK_OW
        n_tiles_oc = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (N, n_tiles_h * n_tiles_w, n_tiles_oc)

        conv2d_fused_kernel[grid](
            x_nhwc, w_nhwc, fused_bias, mult_flat, out,
            N, H, W_, C,
            OH, OW, OC,
            KH, KW,
            BLOCK_OH=BLOCK_OH,
            BLOCK_OW=BLOCK_OW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
            num_stages=2,
        )

        return out