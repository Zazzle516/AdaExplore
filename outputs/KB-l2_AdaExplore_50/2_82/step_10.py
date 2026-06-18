import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_maxpool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC: tl.constexpr, OH, OW,
    PH, PW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    TILE_PH: tl.constexpr,
    TILE_PW: tl.constexpr,
):
    # Each program computes a tile of pooled output:
    #   shape [BLOCK_OC, TILE_PH, TILE_PW]
    # corresponding to (POOL*TILE_PH) x (POOL*TILE_PW) conv outputs
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    num_tiles_w = (PW + TILE_PW - 1) // TILE_PW
    tile_ph_idx = pid_sp // num_tiles_w
    tile_pw_idx = pid_sp % num_tiles_w

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # conv output tile range
    CONV_TH: tl.constexpr = TILE_PH * POOL
    CONV_TW: tl.constexpr = TILE_PW * POOL

    conv_h_start = tile_ph_idx * CONV_TH
    conv_w_start = tile_pw_idx * CONV_TW

    oh_range = conv_h_start + tl.arange(0, CONV_TH)  # [CONV_TH]
    ow_range = conv_w_start + tl.arange(0, CONV_TW)  # [CONV_TW]

    oh_mask = oh_range < OH
    ow_mask = ow_range < OW

    # Compute conv outputs for this tile: [BLOCK_OC, CONV_TH, CONV_TW]
    # Flatten spatial to [CONV_TH * CONV_TW]
    SPAT: tl.constexpr = CONV_TH * CONV_TW
    spat_idx = tl.arange(0, SPAT)
    sh = spat_idx // CONV_TW  # [SPAT]
    sw = spat_idx % CONV_TW   # [SPAT]

    oh = conv_h_start + sh  # [SPAT]
    ow = conv_w_start + sw  # [SPAT]
    spat_valid = (oh < OH) & (ow < OW)  # [SPAT]

    acc = tl.zeros((BLOCK_OC, SPAT), dtype=tl.float32)

    # Convolution accumulation
    for ic in tl.static_range(IC):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = oh + kh  # [SPAT]
                iw = ow + kw  # [SPAT]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=spat_valid, other=0.0)  # [SPAT]

                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[:, None]
    # tanh
    e2x = tl.exp(2.0 * acc)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    # scale + bias
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    scaled = tanh_val * SCALE + bias_vals[:, None]
    # mask invalid positions
    neg_inf = float('-inf')
    scaled = tl.where(spat_valid[None, :], scaled, neg_inf)

    # Reshape into [BLOCK_OC, CONV_TH, CONV_TW] then max-pool to [BLOCK_OC, TILE_PH, TILE_PW]
    scaled = tl.reshape(scaled, (BLOCK_OC, CONV_TH, CONV_TW))
    # Reshape -> [BLOCK_OC, TILE_PH, POOL, TILE_PW, POOL] and reduce over POOL dims
    scaled = tl.reshape(scaled, (BLOCK_OC, TILE_PH, POOL, TILE_PW, POOL))
    # max over axis 4 (last POOL)
    m1 = tl.max(scaled, axis=4)  # [BLOCK_OC, TILE_PH, POOL, TILE_PW]
    m2 = tl.max(m1, axis=2)      # [BLOCK_OC, TILE_PH, TILE_PW]

    # Store
    ph_offs = tile_ph_idx * TILE_PH + tl.arange(0, TILE_PH)  # [TILE_PH]
    pw_offs = tile_pw_idx * TILE_PW + tl.arange(0, TILE_PW)  # [TILE_PW]
    ph_mask = ph_offs < PH
    pw_mask = pw_offs < PW

    P_total = PH * PW
    out_base = pid_n * (OC * P_total)

    # out[n, oc, ph, pw]
    oc_idx = oc_offs[:, None, None]  # [BLOCK_OC,1,1]
    ph_idx = ph_offs[None, :, None]  # [1,TILE_PH,1]
    pw_idx = pw_offs[None, None, :]  # [1,1,TILE_PW]
    out_off = out_base + oc_idx * P_total + ph_idx * PW + pw_idx
    out_mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    tl.store(out_ptr + out_off, m2, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, IH, IW = x.shape
        KH = KW = self.kernel_size
        OC = self.out_channels
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        w = self.conv.weight.contiguous().cuda()
        cb = self.conv.bias.contiguous().cuda()
        bias_flat = self.bias.view(-1).contiguous().cuda()

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 16
        TILE_PH = 4
        TILE_PW = 8

        num_tiles_h = (PH + TILE_PH - 1) // TILE_PH
        num_tiles_w = (PW + TILE_PW - 1) // TILE_PW

        grid = (N, triton.cdiv(OC, BLOCK_OC), num_tiles_h * num_tiles_w)

        fused_conv_tanh_scale_bias_maxpool_kernel[grid](
            x, w, cb, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
            BLOCK_OC, TILE_PH, TILE_PW,
            num_warps=4, num_stages=2,
        )
        return out