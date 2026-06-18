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
    BLOCK_POH: tl.constexpr,
    BLOCK_POW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # program ids
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    num_pow_blocks = tl.cdiv(POW, BLOCK_POW)
    pid_poh = pid_sp // num_pow_blocks
    pid_pow = pid_sp % num_pow_blocks

    # output pool coords for this tile
    poh_offs = pid_poh * BLOCK_POH + tl.arange(0, BLOCK_POH)  # [BLOCK_POH]
    pow_offs = pid_pow * BLOCK_POW + tl.arange(0, BLOCK_POW)  # [BLOCK_POW]
    poh_mask = poh_offs < POH
    pow_mask = pow_offs < POW

    # OC tile
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # We compute the conv at the (BLOCK_POH*PK) x (BLOCK_POW*PK) output region.
    CONV_H: tl.constexpr = BLOCK_POH * PK
    CONV_W: tl.constexpr = BLOCK_POW * PK

    # conv output coords
    oh_offs = pid_poh * BLOCK_POH * PK + tl.arange(0, CONV_H)  # [CONV_H]
    ow_offs = pid_pow * BLOCK_POW * PK + tl.arange(0, CONV_W)  # [CONV_W]
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    # accumulator for conv: [BLOCK_OC, CONV_H, CONV_W]
    # Flatten H,W to reduce dim count
    acc = tl.zeros((BLOCK_OC, CONV_H * CONV_W), dtype=tl.float32)

    # flatten conv output spatial coords
    sp_h = tl.arange(0, CONV_H)[:, None] + tl.zeros((1, CONV_W), dtype=tl.int32)  # [CONV_H, CONV_W]
    sp_w = tl.zeros((CONV_H, 1), dtype=tl.int32) + tl.arange(0, CONV_W)[None, :]
    sp_h_flat = tl.reshape(sp_h, (CONV_H * CONV_W,))
    sp_w_flat = tl.reshape(sp_w, (CONV_H * CONV_W,))

    oh_grid = pid_poh * BLOCK_POH * PK + sp_h_flat  # [CONV_H*CONV_W]
    ow_grid = pid_pow * BLOCK_POW * PK + sp_w_flat
    valid_sp = (oh_grid < OH) & (ow_grid < OW)

    # Loop over input channels and kernel
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_grid = oh_grid + kh  # [SP]
            iw_grid = ow_grid + kw

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                ic_mask = ic_offs < IC

                # weight: w[oc, ic, kh, kw] -> [BLOCK_OC, BLOCK_IC]
                w_off = oc_offs[:, None] * (IC * KH * KW) + ic_offs[None, :] * (KH * KW) + kh * KW + kw
                w_msk = oc_mask[:, None] & ic_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_msk, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                # input: x[n, ic, ih, iw] -> [BLOCK_IC, SP]
                x_off = (pid_n * (IC * IH * IW)
                         + ic_offs[:, None] * (IH * IW)
                         + ih_grid[None, :] * IW
                         + iw_grid[None, :])
                x_msk = ic_mask[:, None] & valid_sp[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_msk, other=0.0)  # [BLOCK_IC, SP]

                acc += tl.dot(w_val, x_val)

    # add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += b_val[:, None]

    # subtract sub1, tanh, subtract sub2
    v = acc - SUB1
    e2x = tl.exp(2.0 * v)
    t = (e2x - 1.0) / (e2x + 1.0)
    v = t - SUB2  # [BLOCK_OC, CONV_H*CONV_W]

    # Reshape to [BLOCK_OC, CONV_H, CONV_W] and pool
    v3 = tl.reshape(v, (BLOCK_OC, CONV_H, CONV_W))
    # Reshape to [BLOCK_OC, BLOCK_POH, PK, BLOCK_POW, PK]
    v5 = tl.reshape(v3, (BLOCK_OC, BLOCK_POH, PK, BLOCK_POW, PK))
    # Sum over PK axes
    s1 = tl.sum(v5, axis=4)  # [BLOCK_OC, BLOCK_POH, PK, BLOCK_POW]
    s2 = tl.sum(s1, axis=2)  # [BLOCK_OC, BLOCK_POH, BLOCK_POW]
    pooled = s2 * INV_PK2

    # Store: out[n, oc, poh, pow]
    out_off = (pid_n * (OC * POH * POW)
               + oc_offs[:, None, None] * (POH * POW)
               + poh_offs[None, :, None] * POW
               + pow_offs[None, None, :])
    out_mask = oc_mask[:, None, None] & poh_mask[None, :, None] & pow_mask[None, None, :]
    tl.store(out_ptr + out_off, pooled, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = int(kernel_size_pool)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

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
        BLOCK_POH = 4
        BLOCK_POW = 8  # CONV tile = 8x16 = 128 spatial elements
        BLOCK_IC = 16

        num_poh_blocks = triton.cdiv(POH, BLOCK_POH)
        num_pow_blocks = triton.cdiv(POW, BLOCK_POW)

        grid = (N, triton.cdiv(OC, BLOCK_OC), num_poh_blocks * num_pow_blocks)

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW, PK,
            self.subtract1_value, self.subtract2_value,
            1.0 / (PK * PK),
            BLOCK_OC, BLOCK_POH, BLOCK_POW, BLOCK_IC,
            num_warps=4, num_stages=2,
        )

        return out