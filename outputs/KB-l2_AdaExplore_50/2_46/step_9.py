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

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # Output base oh, ow per pooled position
    oh_base = poh * POOL  # [BLOCK_SP]
    ow_base = pow_ * POOL  # [BLOCK_SP]

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # For each pool position (ph, pw) accumulate conv result
    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            oh = oh_base + ph
            ow = ow_base + pw

            conv_val = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            # Loop K = IC * KH * KW; tile over IC with BLOCK_IC.
            for ic_base in range(0, IC, BLOCK_IC):
                ic_offs = ic_base + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC
                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = oh + kh
                        iw = ow + kw

                        x_off = (pid_n * IC * IH * IW
                                 + ic_offs[:, None] * (IH * IW)
                                 + ih[None, :] * IW
                                 + iw[None, :])
                        x_mask = ic_mask[:, None] & sp_mask[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                        w_off = (oc_offs[:, None] * (IC * KH * KW)
                                 + ic_offs[None, :] * (KH * KW)
                                 + kh * KW + kw)
                        w_mask = oc_mask[:, None] & ic_mask[None, :]
                        w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                        conv_val += tl.dot(w_val, x_val)

            b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
            conv_val += b_val[:, None]

            v = conv_val - SUB1
            # tanh
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - SUB2
            acc += v

    inv = 1.0 / (POOL * POOL)
    acc = acc * inv

    out_off = (pid_n * (OC * POH * POW)
               + oc_offs[:, None] * (POH * POW)
               + sp_offs[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


@triton.jit
def fused_conv_tanh_pool_kernel_nhwc(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    POH, POW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    """
    Layout:
      x:   NHWC, contiguous as (N, IH, IW, IC)
      w:   (OC, KH, KW, IC), contiguous
      out: NCHW, (N, OC, POH, POW)
    One program per (N, OC tile, POOL_SP tile).
    For each pool position, we compute a [BLOCK_OC, BLOCK_SP] conv output
    via tl.dot over K = IC * KH * KW (tiled by BLOCK_IC over IC).
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    oh_base = poh * POOL
    ow_base = pow_ * POOL

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    inv = 1.0 / (POOL * POOL)

    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            oh = oh_base + ph  # [BLOCK_SP]
            ow = ow_base + pw  # [BLOCK_SP]

            conv_val = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh  # [BLOCK_SP]
                    iw = ow + kw  # [BLOCK_SP]
                    # base offset into x: (n, ih, iw, :) row in NHWC
                    x_base = (pid_n * IH * IW * IC
                              + ih[None, :] * (IW * IC)
                              + iw[None, :] * IC)  # [1, BLOCK_SP]
                    # base offset into w: (oc, kh, kw, :)
                    w_base = (oc_offs[:, None] * (KH * KW * IC)
                              + kh * (KW * IC)
                              + kw * IC)  # [BLOCK_OC, 1]

                    for ic_base in range(0, IC, BLOCK_IC):
                        ic_offs = ic_base + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                        ic_mask = ic_offs < IC

                        # x: [BLOCK_IC, BLOCK_SP]
                        x_off = ic_offs[:, None] + x_base  # [BLOCK_IC, BLOCK_SP]
                        x_mask = ic_mask[:, None] & sp_mask[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                        # w: [BLOCK_OC, BLOCK_IC]
                        w_off = w_base + ic_offs[None, :]
                        w_mask = oc_mask[:, None] & ic_mask[None, :]
                        w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                        conv_val += tl.dot(w_val, x_val)

            b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
            conv_val += b_val[:, None]

            v = conv_val - SUB1
            e2 = tl.exp(2.0 * v)
            t = (e2 - 1.0) / (e2 + 1.0)
            v = t - SUB2
            acc += v

    acc = acc * inv

    out_off = (pid_n * (OC * POH * POW)
               + oc_offs[:, None] * (POH * POW)
               + sp_offs[None, :])
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


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

        # Pre-permute weights to (OC, KH, KW, IC) for NHWC GEMM access
        with torch.no_grad():
            w = self.conv.weight.detach()  # (OC, IC, KH, KW)
            w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)
        self.register_buffer("w_nhwc", w_nhwc.cuda(), persistent=False)
        self.register_buffer("b_buf", self.conv.bias.detach().contiguous().cuda(), persistent=False)

    def forward(self, x):
        x = x.cuda()
        # NHWC input
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, IH, IW, IC)

        N, IH, IW, IC = x_nhwc.shape
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
        BLOCK_SP = 64
        BLOCK_IC = 32

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(POH * POW, BLOCK_SP))

        fused_conv_tanh_pool_kernel_nhwc[grid](
            x_nhwc, self.w_nhwc, self.b_buf, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            BLOCK_OC, BLOCK_SP, BLOCK_IC,
            num_warps=4, num_stages=2,
        )
        return out