import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N,
    IC: tl.constexpr,
    IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    # grid: (N * (PH/BLOCK_PH) * (PW/BLOCK_PW), OC/BLOCK_OC)
    pid_n_sp = tl.program_id(0)
    pid_oc = tl.program_id(1)

    pw_tiles = PW // BLOCK_PW
    ph_tiles = PH // BLOCK_PH
    sp_tiles = ph_tiles * pw_tiles

    n = pid_n_sp // sp_tiles
    sp = pid_n_sp % sp_tiles
    ph_tile = sp // pw_tiles
    pw_tile = sp % pw_tiles

    # OC offsets
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    # Pooled output offsets within tile
    ph_in_tile = tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_in_tile = tl.arange(0, BLOCK_PW)  # [BLOCK_PW]

    ph_offs = ph_tile * BLOCK_PH + ph_in_tile  # [BLOCK_PH]
    pw_offs = pw_tile * BLOCK_PW + pw_in_tile  # [BLOCK_PW]

    # Conv output spatial sizes for this tile
    # Each pooled output covers POOL x POOL conv outputs
    # Conv-output region: rows [ph_tile*BLOCK_PH*POOL ... +BLOCK_PH*POOL),
    #                     cols [pw_tile*BLOCK_PW*POOL ... +BLOCK_PW*POOL)
    OH_TILE: tl.constexpr = BLOCK_PH * POOL
    OW_TILE: tl.constexpr = BLOCK_PW * POOL

    oh_in_tile = tl.arange(0, OH_TILE)  # [OH_TILE]
    ow_in_tile = tl.arange(0, OW_TILE)  # [OW_TILE]

    oh_base = ph_tile * OH_TILE
    ow_base = pw_tile * OW_TILE

    cb = tl.load(cb_ptr + oc_offs)  # [BLOCK_OC]
    bb = tl.load(b_ptr + oc_offs)   # [BLOCK_OC]

    # Compute conv for the entire OH_TILE x OW_TILE region for all BLOCK_OC channels
    # acc: [BLOCK_OC, OH_TILE*OW_TILE]
    SP_TILE: tl.constexpr = OH_TILE * OW_TILE
    acc = tl.zeros((BLOCK_OC, SP_TILE), dtype=tl.float32)

    # Spatial indices flatten
    sp_oh = (tl.arange(0, SP_TILE) // OW_TILE)  # [SP_TILE]
    sp_ow = (tl.arange(0, SP_TILE) % OW_TILE)   # [SP_TILE]

    # Conv loop
    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # input row for each sp: ih = oh_base + sp_oh + kh
                ih = oh_base + sp_oh + kh  # [SP_TILE]
                iw = ow_base + sp_ow + kw  # [SP_TILE]
                x_off = ((n * IC + ic) * IH + ih) * IW + iw  # [SP_TILE]
                x_val = tl.load(x_ptr + x_off)  # [SP_TILE]

                w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw  # [BLOCK_OC]
                w_val = tl.load(w_ptr + w_off)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    acc = acc + cb[:, None]
    # tanh
    e2 = tl.exp(2.0 * acc)
    t = (e2 - 1.0) / (e2 + 1.0)
    v = t * SCALE + bb[:, None]  # [BLOCK_OC, SP_TILE]

    # Now reduce over each POOL x POOL block to get pooled output
    # We have v[oc, oh, ow] in [BLOCK_OC, OH_TILE, OW_TILE]
    # Reshape mentally: oh = ph_local * POOL + ki, ow = pw_local * POOL + kj
    # We want max over (ki, kj) for each (ph_local, pw_local)
    # Output shape: [BLOCK_OC, BLOCK_PH, BLOCK_PW]

    # Loop over POOL window and reduce
    POOLED_SIZE: tl.constexpr = BLOCK_PH * BLOCK_PW
    pooled = tl.full((BLOCK_OC, POOLED_SIZE), -1e30, dtype=tl.float32)

    # local pooled indices
    pl_ph = tl.arange(0, POOLED_SIZE) // BLOCK_PW  # [POOLED_SIZE]
    pl_pw = tl.arange(0, POOLED_SIZE) % BLOCK_PW

    for ki in tl.static_range(0, POOL):
        for kj in tl.static_range(0, POOL):
            # source index in SP_TILE: (pl_ph*POOL+ki)*OW_TILE + (pl_pw*POOL+kj)
            src_idx = (pl_ph * POOL + ki) * OW_TILE + (pl_pw * POOL + kj)
            # gather from v: v[:, src_idx]
            # Use tl.load won't work since v is in registers. We need indexing.
            # Instead, build mask/select. Since src_idx varies per pooled output,
            # we can't directly index v. We need to compute v inline or restructure.
            # Alternative: reshape via arithmetic - use tl.reshape
            pass

    # Use reshape approach
    v_4d = tl.reshape(v, (BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL))
    # max over the two POOL dims
    v_max = tl.max(v_4d, axis=4)  # [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW]
    v_max = tl.max(v_max, axis=2)  # [BLOCK_OC, BLOCK_PH, BLOCK_PW]

    # Store
    out_off = (((n * OC + oc_offs[:, None, None]) * PH + ph_offs[None, :, None]) * PW
               + pw_offs[None, None, :])
    tl.store(out_ptr + out_off, v_max)


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

        # Tile sizes
        BLOCK_OC = 32
        BLOCK_PH = 4
        BLOCK_PW = 8

        # Fallback if shapes don't divide evenly
        if (PH * POOL != OH or PW * POOL != OW
                or PH % BLOCK_PH != 0 or PW % BLOCK_PW != 0
                or OC % BLOCK_OC != 0):
            y = self.conv(x)
            y = torch.tanh(y) * self.scaling_factor + self.bias
            return self.max_pool(y)

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        ph_tiles = PH // BLOCK_PH
        pw_tiles = PW // BLOCK_PW
        oc_tiles = OC // BLOCK_OC

        grid = (N * ph_tiles * pw_tiles, oc_tiles)

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, self.conv.weight, self.conv.bias, bias_flat, out,
            N,
            IC,
            IH, IW,
            OC,
            OH, OW,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC, BLOCK_PH, BLOCK_PW,
            num_warps=4,
            num_stages=2,
        )
        return out