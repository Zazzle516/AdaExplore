import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    PK: tl.constexpr,
    SUB1: tl.constexpr,
    SUB2: tl.constexpr,
    INV_PK2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    n_pw_tiles = tl.cdiv(POW, BLOCK_PW)
    pid_ph = pid_sp // n_pw_tiles
    pid_pw = pid_sp % n_pw_tiles

    # output (post-pool) tile
    ph_offs = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]
    ph_mask = ph_offs < POH
    pw_mask = pw_offs < POW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Pre-pool conv tile spans BLOCK_PH*PK rows by BLOCK_PW*PK cols
    CONV_H: tl.constexpr = BLOCK_PH * PK
    CONV_W: tl.constexpr = BLOCK_PW * PK

    oh_offs = pid_ph * CONV_H + tl.arange(0, CONV_H)  # [CONV_H]
    ow_offs = pid_pw * CONV_W + tl.arange(0, CONV_W)  # [CONV_W]
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    # accumulator: [BLOCK_OC, CONV_H, CONV_W]  flattened to [BLOCK_OC, CONV_H*CONV_W]
    acc = tl.zeros((BLOCK_OC, CONV_H * CONV_W), dtype=tl.float32)

    # spatial offsets in the conv-output tile
    sp_h = tl.arange(0, CONV_H)
    sp_w = tl.arange(0, CONV_W)
    sp_mask_2d = (oh_offs[:, None] < OH) & (ow_offs[None, :] < OW)
    sp_mask_flat = tl.reshape(sp_mask_2d, (CONV_H * CONV_W,))

    # Loop over IC, KH, KW
    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # ih, iw for the conv input
                ih = oh_offs + kh  # [CONV_H]
                iw = ow_offs + kw  # [CONV_W]

                # input tile [CONV_H, CONV_W]
                x_off = pid_n * (IC * IH * IW) + ic * (IH * IW) + ih[:, None] * IW + iw[None, :]
                x_m = (ih[:, None] < IH) & (iw[None, :] < IW)
                x_tile = tl.load(x_ptr + x_off, mask=x_m, other=0.0)  # [CONV_H, CONV_W]
                x_flat = tl.reshape(x_tile, (CONV_H * CONV_W,))  # [CONV_H*CONV_W]

                # weights for this (kh, kw, ic) across oc: [BLOCK_OC]
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None] * x_flat[None, :]

    # bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    # subtract1, tanh, subtract2
    v = acc - SUB1
    e2x = tl.exp(2.0 * v)
    t = (e2x - 1.0) / (e2x + 1.0)
    v = t - SUB2

    # reshape to [BLOCK_OC, CONV_H, CONV_W]
    v3 = tl.reshape(v, (BLOCK_OC, CONV_H, CONV_W))

    # average pooling over PK x PK windows -> [BLOCK_OC, BLOCK_PH, BLOCK_PW]
    # Reshape: [BLOCK_OC, BLOCK_PH, PK, BLOCK_PW, PK] then sum over the two PK dims
    v5 = tl.reshape(v3, (BLOCK_OC, BLOCK_PH, PK, BLOCK_PW, PK))
    pooled = tl.sum(v5, axis=4)        # [BLOCK_OC, BLOCK_PH, PK, BLOCK_PW]
    pooled = tl.sum(pooled, axis=2)    # [BLOCK_OC, BLOCK_PH, BLOCK_PW]
    pooled = pooled * INV_PK2

    # store
    out_off = (pid_n * (OC * POH * POW)
               + oc_offs[:, None, None] * (POH * POW)
               + ph_offs[None, :, None] * POW
               + pw_offs[None, None, :])
    out_mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    tl.store(out_ptr + out_off, pooled, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.bias = nn.Parameter(conv.bias.detach().clone())
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def forward(self, x):
        x = x.contiguous()
        if not x.is_cuda:
            x = x.cuda()
        w = self.weight
        b = self.bias
        if not w.is_cuda:
            w = w.cuda()
            b = b.cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        PK = self.kernel_size_pool
        POH = OH // PK
        POW = OW // PK

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_PH = 8
        BLOCK_PW = 16
        BLOCK_IC = 1

        grid = (
            N,
            triton.cdiv(OC, BLOCK_OC),
            triton.cdiv(POH, BLOCK_PH) * triton.cdiv(POW, BLOCK_PW),
        )

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW, PK,
            self.subtract1_value, self.subtract2_value,
            1.0 / (PK * PK),
            BLOCK_OC, BLOCK_PH, BLOCK_PW, BLOCK_IC,
            num_warps=4, num_stages=2,
        )

        return out