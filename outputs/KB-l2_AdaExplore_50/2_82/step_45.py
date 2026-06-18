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
    BLOCK_OC: tl.constexpr,
    TILE_H: tl.constexpr,
    TILE_W: tl.constexpr,
    IC_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    n_tw = (POW + TILE_W - 1) // TILE_W
    pid_th = pid_sp // n_tw
    pid_tw = pid_sp % n_tw

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Output pool tile range
    poh_start = pid_th * TILE_H
    pow_start = pid_tw * TILE_W

    # Conv-output tile range
    oh_start = poh_start * POOL
    ow_start = pow_start * POOL

    # full conv output sub-tile dims
    CH = TILE_H * POOL
    CW = TILE_W * POOL

    # offsets within the conv-output tile
    h_range = tl.arange(0, TILE_H * POOL)  # [CH]
    w_range = tl.arange(0, TILE_W * POOL)  # [CW]

    # accumulator for conv-output tile per OC: [BLOCK_OC, CH, CW]
    # Flattened to [BLOCK_OC, CH*CW]
    SP = (TILE_H * POOL) * (TILE_W * POOL)
    acc = tl.zeros((BLOCK_OC, SP), dtype=tl.float32)

    # Build per-spatial-out (h,w) flattened indices
    sp_h = h_range[:, None] + tl.zeros((1, TILE_W * POOL), dtype=tl.int32)  # [CH, CW]
    sp_w = tl.zeros((TILE_H * POOL, 1), dtype=tl.int32) + w_range[None, :]
    sp_h_flat = tl.reshape(sp_h, (SP,))
    sp_w_flat = tl.reshape(sp_w, (SP,))

    oh = oh_start + sp_h_flat  # [SP]
    ow = ow_start + sp_w_flat  # [SP]
    sp_valid = (oh < OH) & (ow < OW)

    # Conv loop
    for ic in tl.static_range(0, IC_CONST):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=sp_valid, other=0.0)  # [SP]
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    conv_b = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    extra_b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)      # [BLOCK_OC]
    acc = acc + conv_b[:, None]

    # tanh
    e_pos = tl.exp(acc)
    e_neg = tl.exp(-acc)
    t = (e_pos - e_neg) / (e_pos + e_neg)
    t = t * SCALE + extra_b[:, None]
    # Mask invalid positions
    NEG_INF = float(-1e30)
    t = tl.where(sp_valid[None, :], t, NEG_INF)

    # Reshape to [BLOCK_OC, TILE_H, POOL, TILE_W, POOL] -> max over POOL,POOL
    t_resh = tl.reshape(t, (BLOCK_OC, TILE_H, POOL, TILE_W, POOL))
    # max over last (POOL) -> [BLOCK_OC, TILE_H, POOL, TILE_W]
    t1 = tl.max(t_resh, axis=4)
    # max over POOL axis (now axis=2) -> [BLOCK_OC, TILE_H, TILE_W]
    t2 = tl.max(t1, axis=2)

    # Store
    # out indices
    th_range = tl.arange(0, TILE_H)
    tw_range = tl.arange(0, TILE_W)
    poh_idx = poh_start + th_range  # [TILE_H]
    pow_idx = pow_start + tw_range  # [TILE_W]
    poh_mask = poh_idx < POH
    pow_mask = pow_idx < POW

    # Flatten [BLOCK_OC, TILE_H, TILE_W]
    out_base = pid_n * (OC * POH * POW)
    # out[n, oc, ph, pw] = out_ptr + out_base + oc*POH*POW + ph*POW + pw
    oc_stride = POH * POW
    out_off = (
        out_base
        + oc_offs[:, None, None] * oc_stride
        + poh_idx[None, :, None] * POW
        + pow_idx[None, None, :]
    )
    out_mask = oc_mask[:, None, None] & poh_mask[None, :, None] & pow_mask[None, None, :]
    tl.store(out_ptr + out_off, t2, mask=out_mask)


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

        BLOCK_OC = 32
        TILE_H = 2
        TILE_W = 4

        n_th = (POH + TILE_H - 1) // TILE_H
        n_tw = (POW + TILE_W - 1) // TILE_W

        grid = (N, triton.cdiv(OC, BLOCK_OC), n_th * n_tw)

        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, w, cb, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC, TILE_H, TILE_W,
            IC,
            num_warps=8, num_stages=2,
        )
        return out