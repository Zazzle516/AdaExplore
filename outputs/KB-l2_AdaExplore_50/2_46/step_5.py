import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC,
    IH: tl.constexpr, IW: tl.constexpr,
    OC,
    OH: tl.constexpr, OW: tl.constexpr,
    POH: tl.constexpr, POW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    pool_idx = tl.arange(0, POOL * POOL)
    ph = pool_idx // POOL
    pw = pool_idx % POOL

    oh = poh[:, None] * POOL + ph[None, :]
    ow = pow_[:, None] * POOL + pw[None, :]

    SP_FULL: tl.constexpr = BLOCK_SP * POOL * POOL
    oh_flat = tl.reshape(oh, (SP_FULL,))
    ow_flat = tl.reshape(ow, (SP_FULL,))

    sp_mask_pooled = sp_offs < (POH * POW)
    sp_mask_full = tl.reshape(tl.broadcast_to(sp_mask_pooled[:, None], (BLOCK_SP, POOL * POOL)), (SP_FULL,))

    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC, SP_FULL), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih_flat = oh_flat + kh
            iw_flat = ow_flat + kw
            sp_in_off = ih_flat * IW + iw_flat

            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                x_off = pid_n * (IC * IH * IW) + ic_offs[:, None] * (IH * IW) + sp_in_off[None, :]
                x_mask = ic_mask[:, None] & sp_mask_full[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                w_off = oc_offs[:, None] * (IC * KH * KW) + ic_offs[None, :] * (KH * KW) + kh * KW + kw
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc = tl.dot(w_val, x_val, acc)

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + b_val[:, None]

    v = acc - SUB1
    e2 = tl.exp(2.0 * v)
    t = (e2 - 1.0) / (e2 + 1.0)
    v = t - SUB2

    v_r = tl.reshape(v, (BLOCK_OC, BLOCK_SP, POOL * POOL))
    pooled = tl.sum(v_r, axis=2) / (POOL * POOL)

    out_off = pid_n * (OC * POH * POW) + oc_offs[:, None] * (POH * POW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask_pooled[None, :]
    tl.store(out_ptr + out_off, pooled, mask=mask)


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
        POOL = self.kernel_size_pool
        POH = OH // POOL
        POW = OW // POOL

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_SP = 64  # pooled spatial; real = 64*4 = 256
        BLOCK_IC = 32

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_SP))

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC,
            IH, IW,
            OC,
            OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            BLOCK_OC, BLOCK_SP, BLOCK_IC,
            num_warps=8, num_stages=3,
        )
        return out