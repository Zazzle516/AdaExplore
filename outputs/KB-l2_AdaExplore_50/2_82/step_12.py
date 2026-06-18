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
    TILE_PH: tl.constexpr,  # pooled tile height
    TILE_PW: tl.constexpr,  # pooled tile width
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_tiles_w = (PW + TILE_PW - 1) // TILE_PW
    tile_ph_idx = pid_p // num_tiles_w
    tile_pw_idx = pid_p % num_tiles_w

    # conv-output tile size (covers an integer number of pool cells)
    CONV_TH: tl.constexpr = TILE_PH * POOL
    CONV_TW: tl.constexpr = TILE_PW * POOL

    # starting conv-output coords for this tile
    oh_start = tile_ph_idx * CONV_TH
    ow_start = tile_pw_idx * CONV_TW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # conv output grid offsets within tile
    th = tl.arange(0, CONV_TH)  # [CONV_TH]
    tw = tl.arange(0, CONV_TW)  # [CONV_TW]
    oh = oh_start + th  # [CONV_TH]
    ow = ow_start + tw  # [CONV_TW]
    valid_h = oh < OH  # [CONV_TH]
    valid_w = ow < OW  # [CONV_TW]
    # [CONV_TH, CONV_TW]
    valid_hw = valid_h[:, None] & valid_w[None, :]

    # Accumulator for conv: [BLOCK_OC, CONV_TH * CONV_TW] flattened, but
    # we keep it 3D-flat as [BLOCK_OC, CONV_TH*CONV_TW]
    CT: tl.constexpr = CONV_TH * CONV_TW
    acc = tl.zeros((BLOCK_OC, CT), dtype=tl.float32)

    # Flatten output coords for index math
    oh_flat = tl.arange(0, CT) // CONV_TW  # [CT] -> th index
    ow_flat = tl.arange(0, CT) % CONV_TW   # [CT] -> tw index
    oh_g = oh_start + oh_flat  # global oh
    ow_g = ow_start + ow_flat  # global ow
    valid_flat = (oh_g < OH) & (ow_g < OW)  # [CT]

    # iterate over IC, KH, KW
    for ic in tl.static_range(IC):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = oh_g + kh  # [CT]
                iw = ow_g + kw  # [CT]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih * IW + iw  # [CT]
                x_val = tl.load(x_ptr + x_off, mask=valid_flat, other=0.0)  # [CT]
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw  # [BLOCK_OC]
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + cb[:, None]

    # tanh, scale, add bias
    e2x = tl.exp(2.0 * acc)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    scaled = tanh_val * SCALE + bias_vals[:, None]

    neg_inf = float('-inf')
    scaled = tl.where(valid_flat[None, :], scaled, neg_inf)

    # Now max-pool: reshape [BLOCK_OC, CONV_TH, CONV_TW] -> reduce over POOL x POOL
    scaled = tl.reshape(scaled, (BLOCK_OC, TILE_PH, POOL, TILE_PW, POOL))
    # reduce over POOL dims (axes 2 and 4)
    pooled = tl.max(scaled, axis=4)  # [BLOCK_OC, TILE_PH, POOL, TILE_PW]
    pooled = tl.max(pooled, axis=2)  # [BLOCK_OC, TILE_PH, TILE_PW]

    # Store
    p_h_base = tile_ph_idx * TILE_PH
    p_w_base = tile_pw_idx * TILE_PW
    p_h_off = p_h_base + tl.arange(0, TILE_PH)  # [TILE_PH]
    p_w_off = p_w_base + tl.arange(0, TILE_PW)  # [TILE_PW]
    p_h_mask = p_h_off < PH
    p_w_mask = p_w_off < PW

    P_total = PH * PW
    # out[n, oc, p_h, p_w]
    out_base = pid_n * (OC * P_total) + oc_offs[:, None, None] * P_total \
        + p_h_off[None, :, None] * PW + p_w_off[None, None, :]
    out_mask = oc_mask[:, None, None] & p_h_mask[None, :, None] & p_w_mask[None, None, :]
    tl.store(out_ptr + out_base, pooled, mask=out_mask)


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