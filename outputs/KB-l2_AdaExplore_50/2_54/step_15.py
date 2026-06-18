import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv2d_fused_kernel(
    x_ptr,       # NHWC input [N, H, W, IC]
    w_ptr,       # weights [OC, KH*KW*IC]
    bias_ptr,    # fused bias [OC] = conv_bias * multiplier
    scale_ptr,   # multiplier [OC]
    out_ptr,     # NCHW output [N, OC, OH, OW]
    N, H, W, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_spatial = tl.program_id(0)  # over (N * tiles_oh * tiles_ow)
    pid_oc = tl.program_id(1)       # over OC tiles

    tiles_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
    tiles_oh = (OH + BLOCK_OH - 1) // BLOCK_OH
    tiles_per_n = tiles_oh * tiles_ow

    n_idx = pid_spatial // tiles_per_n
    rem = pid_spatial % tiles_per_n
    tile_oh = rem // tiles_ow
    tile_ow = rem % tiles_ow

    oh_start = tile_oh * BLOCK_OH
    ow_start = tile_ow * BLOCK_OW
    oc_start = pid_oc * BLOCK_OC

    offs_oh = oh_start + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    offs_ow = ow_start + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    offs_oc = oc_start + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    oh_mask = offs_oh < OH
    ow_mask = offs_ow < OW
    oc_mask = offs_oc < OC

    # Spatial linearized: [BLOCK_OH * BLOCK_OW]
    spatial_idx = tl.arange(0, BLOCK_OH * BLOCK_OW)
    sp_oh = spatial_idx // BLOCK_OW
    sp_ow = spatial_idx % BLOCK_OW
    cur_oh = oh_start + sp_oh  # [BLOCK_TILE]
    cur_ow = ow_start + sp_ow  # [BLOCK_TILE]
    sp_mask = (cur_oh < OH) & (cur_ow < OW)  # [BLOCK_TILE]

    acc = tl.zeros((BLOCK_OH * BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    K = KH * KW * IC

    # Loop kh, kw, ic explicitly
    for kh in tl.static_range(0, KH):
        ih = cur_oh + kh  # [BLOCK_TILE]
        for kw in tl.static_range(0, KW):
            iw = cur_ow + kw  # [BLOCK_TILE]
            # ic loop
            for ic_start in range(0, IC, BLOCK_IC):
                offs_ic = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                ic_mask = offs_ic < IC

                # x[n, ih, iw, ic]
                x_offs = (n_idx * (H * W * IC)
                          + ih[:, None] * (W * IC)
                          + iw[:, None] * IC
                          + offs_ic[None, :])
                x_m = sp_mask[:, None] & ic_mask[None, :]
                x_tile = tl.load(x_ptr + x_offs, mask=x_m, other=0.0)  # [BLOCK_TILE, BLOCK_IC]

                # w[oc, kh, kw, ic] -> flattened k index
                k_idx = kh * (KW * IC) + kw * IC + offs_ic  # [BLOCK_IC]
                w_offs = offs_oc[:, None] * K + k_idx[None, :]  # [BLOCK_OC, BLOCK_IC]
                w_m = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptr + w_offs, mask=w_m, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                acc += tl.dot(x_tile, tl.trans(w_tile))

    # Epilogue
    scale = tl.load(scale_ptr + offs_oc, mask=oc_mask, other=0.0)
    bias = tl.load(bias_ptr + offs_oc, mask=oc_mask, other=0.0)

    y = acc * scale[None, :] + bias[None, :]
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.7071067811865475
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    # Store to NCHW: out[n, oc, oh, ow]
    # acc shape [BLOCK_TILE, BLOCK_OC] => for each (sp, oc) store
    out_offs = (n_idx * (OC * OH * OW)
                + offs_oc[None, :] * (OH * OW)
                + cur_oh[:, None] * OW
                + cur_ow[:, None])
    store_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Pre-permute weight to NHWC kernel layout [OC, KH, KW, IC] -> [OC, KH*KW*IC]
        self._cached_weight = None
        self._cached_fused_bias = None

    def _prepare(self):
        OC = self.out_channels
        IC = self.in_channels
        KH = self.kernel_size
        KW = self.kernel_size
        w = self.conv.weight  # [OC, IC, KH, KW]
        w_nhwc = w.permute(0, 2, 3, 1).contiguous().view(OC, KH * KW * IC)
        mult_flat = self.multiplier.contiguous().view(-1)
        fused_bias = self.conv.bias * mult_flat
        return w_nhwc, fused_bias, mult_flat

    def forward(self, x):
        N, C, H, W_ = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = H - KH + 1
        OW = W_ - KW + 1

        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_nhwc, fused_bias, mult_flat = self._prepare()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OH = 8
        BLOCK_OW = 32
        BLOCK_OC = 64
        BLOCK_IC = 32

        tiles_oh = (OH + BLOCK_OH - 1) // BLOCK_OH
        tiles_ow = (OW + BLOCK_OW - 1) // BLOCK_OW
        tiles_oc = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (N * tiles_oh * tiles_ow, tiles_oc)

        conv2d_fused_kernel[grid](
            x_nhwc, w_nhwc, fused_bias, mult_flat, out,
            N, H, W_, C,
            OH, OW, OC,
            KH, KW,
            BLOCK_OH=BLOCK_OH,
            BLOCK_OW=BLOCK_OW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_IC=BLOCK_IC,
            num_warps=8,
            num_stages=2,
        )
        return out