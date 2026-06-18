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
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    num_tiles_w = (PW + TILE_PW - 1) // TILE_PW
    tile_ph_idx = pid_sp // num_tiles_w
    tile_pw_idx = pid_sp % num_tiles_w

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    CONV_TH: tl.constexpr = TILE_PH * POOL
    CONV_TW: tl.constexpr = TILE_PW * POOL

    conv_h_start = tile_ph_idx * CONV_TH
    conv_w_start = tile_pw_idx * CONV_TW

    SPAT: tl.constexpr = CONV_TH * CONV_TW
    spat_idx = tl.arange(0, SPAT)
    sh = spat_idx // CONV_TW
    sw = spat_idx % CONV_TW

    oh = conv_h_start + sh
    ow = conv_w_start + sw
    spat_valid = (oh < OH) & (ow < OW)

    acc = tl.zeros((BLOCK_OC, SPAT), dtype=tl.float32)

    x_base = pid_n * (IC * IH * IW)

    for ic in tl.static_range(IC):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = oh + kh
                iw = ow + kw
                x_off = x_base + ic * (IH * IW) + ih * IW + iw
                x_val = tl.load(x_ptr + x_off, mask=spat_valid, other=0.0)

                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                acc += w_val[:, None] * x_val[None, :]

    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + cb[:, None]
    # tanh via fast formulation: 1 - 2/(exp(2x)+1)
    e2x = tl.exp(2.0 * acc)
    tanh_val = 1.0 - 2.0 / (e2x + 1.0)
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    scaled = tanh_val * SCALE + bias_vals[:, None]
    neg_inf = float('-inf')
    scaled = tl.where(spat_valid[None, :], scaled, neg_inf)

    scaled = tl.reshape(scaled, (BLOCK_OC, TILE_PH, POOL, TILE_PW, POOL))
    m1 = tl.max(scaled, axis=4)
    m2 = tl.max(m1, axis=2)

    ph_offs = tile_ph_idx * TILE_PH + tl.arange(0, TILE_PH)
    pw_offs = tile_pw_idx * TILE_PW + tl.arange(0, TILE_PW)
    ph_mask = ph_offs < PH
    pw_mask = pw_offs < PW

    P_total = PH * PW
    out_base = pid_n * (OC * P_total)

    oc_idx = oc_offs[:, None, None]
    ph_idx = ph_offs[None, :, None]
    pw_idx = pw_offs[None, None, :]
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

        BLOCK_OC = 32
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
            num_warps=8, num_stages=2,
        )
        return out