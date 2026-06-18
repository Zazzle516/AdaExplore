import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_scale_bias_maxpool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    TILE_PH: tl.constexpr,
    TILE_PW: tl.constexpr,
    K_PAD: tl.constexpr,  # padded K dim (IC*KH*KW padded to power of 2)
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_tiles_w: tl.constexpr = (PW + TILE_PW - 1) // TILE_PW
    tile_ph_idx = pid_p // num_tiles_w
    tile_pw_idx = pid_p % num_tiles_w

    CONV_TH: tl.constexpr = TILE_PH * POOL
    CONV_TW: tl.constexpr = TILE_PW * POOL
    CT: tl.constexpr = CONV_TH * CONV_TW

    oh_start = tile_ph_idx * CONV_TH
    ow_start = tile_pw_idx * CONV_TW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    # output spatial coordinates
    t_idx = tl.arange(0, CT)
    th_idx = t_idx // CONV_TW
    tw_idx = t_idx % CONV_TW
    oh_g = oh_start + th_idx
    ow_g = ow_start + tw_idx

    K: tl.constexpr = IC * KH * KW

    # K-dim layout: ic * (KH*KW) + kh * KW + kw
    k_idx = tl.arange(0, K_PAD)
    k_mask = k_idx < K
    ic_k = k_idx // (KH * KW)
    rem = k_idx % (KH * KW)
    kh_k = rem // KW
    kw_k = rem % KW

    # Weight: [OC, IC, KH, KW] indexed by (oc, k) -> oc * K + k
    # Load weight tile [BLOCK_OC, K_PAD]
    w_off = oc_offs[:, None] * K + k_idx[None, :]
    w_val = tl.load(w_ptr + w_off, mask=(oc_offs[:, None] < OC) & k_mask[None, :], other=0.0)

    # Build x tile [K_PAD, CT]
    # ih = oh_g + kh, iw = ow_g + kw
    ih = oh_g[None, :] + kh_k[:, None]  # [K_PAD, CT]
    iw = ow_g[None, :] + kw_k[:, None]  # [K_PAD, CT]
    x_off = pid_n * (IC * IH * IW) + ic_k[:, None] * (IH * IW) + ih * IW + iw
    x_val = tl.load(x_ptr + x_off, mask=k_mask[:, None], other=0.0)

    # GEMM: [BLOCK_OC, K_PAD] @ [K_PAD, CT] -> [BLOCK_OC, CT]
    acc = tl.dot(w_val, x_val)

    # Add conv bias
    cb = tl.load(cb_ptr + oc_offs, mask=oc_offs < OC, other=0.0)
    acc = acc + cb[:, None]

    # tanh
    e2x = tl.exp(2.0 * acc)
    tanh_val = (e2x - 1.0) / (e2x + 1.0)

    # scale + bias
    bias_vals = tl.load(b_ptr + oc_offs, mask=oc_offs < OC, other=0.0)
    scaled = tanh_val * SCALE + bias_vals[:, None]

    # Max pool: reshape [BLOCK_OC, TILE_PH, POOL, TILE_PW, POOL]
    scaled = tl.reshape(scaled, (BLOCK_OC, TILE_PH, POOL, TILE_PW, POOL))
    pooled = tl.max(scaled, axis=4)
    pooled = tl.max(pooled, axis=2)  # [BLOCK_OC, TILE_PH, TILE_PW]

    # Store
    p_h_base = tile_ph_idx * TILE_PH
    p_w_base = tile_pw_idx * TILE_PW
    p_h_off = p_h_base + tl.arange(0, TILE_PH)
    p_w_off = p_w_base + tl.arange(0, TILE_PW)

    P_total: tl.constexpr = PH * PW
    out_base = pid_n * (OC * P_total) + oc_offs[:, None, None] * P_total \
        + p_h_off[None, :, None] * PW + p_w_off[None, None, :]
    out_mask = (oc_offs[:, None, None] < OC) & (p_h_off[None, :, None] < PH) & (p_w_off[None, None, :] < PW)
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

        BLOCK_OC = 64
        TILE_PH = 4
        TILE_PW = 4

        K = IC * KH * KW
        # pad K to next power of two >= 16 for tl.dot
        K_PAD = 1
        while K_PAD < max(K, 16):
            K_PAD *= 2

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
            K_PAD,
            num_warps=4, num_stages=3,
        )
        return out