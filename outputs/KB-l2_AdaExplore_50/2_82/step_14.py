import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    TILE_PH: tl.constexpr,
    TILE_PW: tl.constexpr,
    IC_C: tl.constexpr,
):
    # grid: (N, OC // BLOCK_OC, (PH/TILE_PH) * (PW/TILE_PW))
    n = tl.program_id(0)
    oc_blk = tl.program_id(1)
    tile = tl.program_id(2)

    n_tiles_w = PW // TILE_PW
    tile_ph = tile // n_tiles_w
    tile_pw = tile % n_tiles_w

    OH_TILE: tl.constexpr = TILE_PH * POOL
    OW_TILE: tl.constexpr = TILE_PW * POOL
    IH_TILE: tl.constexpr = OH_TILE + KH - 1
    IW_TILE: tl.constexpr = OW_TILE + KW - 1

    oc_offs = oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)

    oh_origin = tile_ph * OH_TILE
    ow_origin = tile_pw * OW_TILE

    ih_range = oh_origin + tl.arange(0, IH_TILE)  # [IH_TILE]
    iw_range = ow_origin + tl.arange(0, IW_TILE)  # [IW_TILE]

    cb = tl.load(cb_ptr + oc_offs)
    bb = tl.load(b_ptr + oc_offs)

    # Accumulator: [BLOCK_OC, OH_TILE*OW_TILE]
    acc = tl.zeros((BLOCK_OC, OH_TILE * OW_TILE), dtype=tl.float32)

    # Output coordinate grids (relative)
    out_rows = tl.arange(0, OH_TILE)  # [OH_TILE]
    out_cols = tl.arange(0, OW_TILE)  # [OW_TILE]

    for ic in tl.static_range(0, IC_C):
        # Load input patch [IH_TILE, IW_TILE]
        in_offsets = ((n * IC + ic) * IH + ih_range[:, None]) * IW + iw_range[None, :]
        x_patch = tl.load(x_ptr + in_offsets)  # [IH_TILE, IW_TILE]

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # weight slice [BLOCK_OC]
                w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw
                w_val = tl.load(w_ptr + w_off)  # [BLOCK_OC]

                # extract sub-patch [OH_TILE, OW_TILE] from x_patch
                rows = kh + out_rows  # [OH_TILE]
                cols = kw + out_cols  # [OW_TILE]
                # use gather via flat index
                idx = rows[:, None] * IW_TILE + cols[None, :]  # [OH_TILE, OW_TILE]
                x_patch_flat = tl.reshape(x_patch, (IH_TILE * IW_TILE,))
                sub = tl.load(x_patch_flat + idx) if False else None
                # Triton: can't do that, recompute via direct load is safest
                sub = tl.load(
                    x_ptr
                    + ((n * IC + ic) * IH + (oh_origin + rows[:, None])) * IW
                    + (ow_origin + cols[None, :])
                )  # [OH_TILE, OW_TILE]
                sub_flat = tl.reshape(sub, (OH_TILE * OW_TILE,))
                acc += w_val[:, None] * sub_flat[None, :]

    acc = acc + cb[:, None]
    # tanh stable
    e2 = tl.exp(-2.0 * tl.abs(acc))
    t = tl.where(acc >= 0, (1.0 - e2) / (1.0 + e2), -(1.0 - e2) / (1.0 + e2))
    v = t * SCALE + bb[:, None]

    v3 = tl.reshape(v, (BLOCK_OC, OH_TILE, OW_TILE))
    v5 = tl.reshape(v3, (BLOCK_OC, TILE_PH, POOL, TILE_PW, POOL))
    pooled = tl.max(tl.max(v5, axis=4), axis=2)  # [BLOCK_OC, TILE_PH, TILE_PW]

    ph_idx = tile_ph * TILE_PH + tl.arange(0, TILE_PH)
    pw_idx = tile_pw * TILE_PW + tl.arange(0, TILE_PW)
    out_offsets = ((n * OC + oc_offs[:, None, None]) * PH + ph_idx[None, :, None]) * PW + pw_idx[None, None, :]
    tl.store(out_ptr + out_offsets, pooled)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        if PH * POOL != OH or PW * POOL != OW:
            y = self.conv(x)
            y = torch.tanh(y) * self.scaling_factor + self.bias
            return self.max_pool(y)

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 32
        TILE_PH = 2
        TILE_PW = 4

        if (PH % TILE_PH != 0) or (PW % TILE_PW != 0) or (OC % BLOCK_OC != 0):
            # Try simpler tile
            TILE_PH = 1
            TILE_PW = 1
            if (OC % BLOCK_OC != 0):
                y = self.conv(x)
                y = torch.tanh(y) * self.scaling_factor + self.bias
                return self.max_pool(y)

        n_tiles = (PH // TILE_PH) * (PW // TILE_PW)
        grid = (N, OC // BLOCK_OC, n_tiles)
        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, self.conv.weight, self.conv.bias, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC, TILE_PH, TILE_PW,
            IC,
            num_warps=8,
            num_stages=3,
        )
        return out