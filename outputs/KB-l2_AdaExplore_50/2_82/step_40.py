import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    n_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW
    pid_ph = pid_sp // n_pw_tiles
    pid_pw = pid_sp % n_pw_tiles

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # Pool tile spans BLOCK_PH x BLOCK_PW pool windows.
    # Each pool window is POOL x POOL conv outputs.
    # Total conv outputs to compute: (BLOCK_PH * POOL) x (BLOCK_PW * POOL)
    OH_TILE: tl.constexpr = BLOCK_PH * POOL
    OW_TILE: tl.constexpr = BLOCK_PW * POOL

    oh_base = pid_ph * OH_TILE
    ow_base = pid_pw * OW_TILE

    oh_local = tl.arange(0, OH_TILE)  # [OH_TILE]
    ow_local = tl.arange(0, OW_TILE)  # [OW_TILE]
    oh = oh_base + oh_local
    ow = ow_base + ow_local

    oh_mask = oh < OH
    ow_mask = ow < OW

    # Accumulator [BLOCK_OC, OH_TILE, OW_TILE]
    acc = tl.zeros((BLOCK_OC, OH_TILE * OW_TILE), dtype=tl.float32)

    # Load conv bias and extra bias
    conv_b = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    extra_b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    # spatial flat indexing
    sp_h = tl.arange(0, OH_TILE)[:, None]  # [OH_TILE, 1]
    sp_w = tl.arange(0, OW_TILE)[None, :]  # [1, OW_TILE]
    sp_h_flat = tl.reshape(sp_h + tl.zeros((OH_TILE, OW_TILE), dtype=tl.int32), (OH_TILE * OW_TILE,))
    sp_w_flat = tl.reshape(sp_w + tl.zeros((OH_TILE, OW_TILE), dtype=tl.int32), (OH_TILE * OW_TILE,))

    oh_flat = oh_base + sp_h_flat  # [OH_TILE*OW_TILE]
    ow_flat = ow_base + sp_w_flat

    sp_mask = (oh_flat < OH) & (ow_flat < OW)

    # K loop over IC * KH * KW - fully unrolled (IC_C is compile-time constant)
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_flat + kh
                iw = ow_flat + kw
                x_off = pid_n * (IC_C * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=sp_mask, other=0.0)  # [SP]
                w_off = oc_offs * (IC_C * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += w_val[:, None] * x_val[None, :]

    # add conv bias
    acc = acc + conv_b[:, None]
    # fast tanh: 2*sigmoid(2x) - 1
    two_x = acc * 2.0
    t = 2.0 / (1.0 + tl.exp(-two_x)) - 1.0
    t = t * SCALE + extra_b[:, None]

    # Reshape to [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL]
    t2 = tl.reshape(t, (BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL))
    # Reduce over POOL axes
    pooled = tl.max(t2, axis=4)   # [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW]
    pooled = tl.max(pooled, axis=2)  # [BLOCK_OC, BLOCK_PH, BLOCK_PW]

    # Store
    ph_offs = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]
    ph_mask = ph_offs < POH
    pw_mask = pw_offs < POW

    out_off = (pid_n * (OC * POH * POW)
               + oc_offs[:, None, None] * (POH * POW)
               + ph_offs[None, :, None] * POW
               + pw_offs[None, None, :])
    out_mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    tl.store(out_ptr + out_off, pooled, mask=out_mask)


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
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        POH = OH // POOL
        POW = OW // POOL

        w = self.conv.weight.contiguous()
        cb = self.conv.bias.contiguous()
        b = self.bias.view(-1).contiguous()

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_PH = 2
        BLOCK_PW = 2

        n_ph_tiles = (POH + BLOCK_PH - 1) // BLOCK_PH
        n_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW

        grid = (N, triton.cdiv(OC, BLOCK_OC), n_ph_tiles * n_pw_tiles)

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, w, cb, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.scaling_factor,
            IC,
            BLOCK_OC, BLOCK_PH, BLOCK_PW,
            num_warps=4, num_stages=2,
        )
        return out