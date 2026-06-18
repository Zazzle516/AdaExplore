import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    ID, IH, IW,
    CD, CH, CW,
    OD, OH, OW,
    IC: tl.constexpr,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow)
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    cd_base = od * P
    ch_base = oh * P
    cw_base = ow * P

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    max_vals = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)

    x_n_base = n * IC * ID * IH * IW
    KVOL = IC * KD * KH * KW

    # Pre-fetch all weights into registers
    # weight layout: (OC, IC, KD, KH, KW), flattened K = IC*KD*KH*KW
    # We'll do per-window loop, but cache the partial conv values via tile load

    for di in tl.static_range(P):
        for hi in tl.static_range(P):
            for wi in tl.static_range(P):
                cd = cd_base + di
                ch = ch_base + hi
                cw = cw_base + wi

                acc = bias

                for ic in tl.static_range(IC):
                    for kd in tl.static_range(KD):
                        for kh in tl.static_range(KH):
                            for kw in tl.static_range(KW):
                                id_ = cd + kd
                                ih_ = ch + kh
                                iw_ = cw + kw
                                x_off = x_n_base + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                                xv = tl.load(x_ptr + x_off)
                                w_off = oc_offs * KVOL + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc = acc + xv * wv

                m = tl.max(tl.where(oc_mask, acc, -float('inf')), axis=0)
                ex = tl.exp(acc - m)
                ex = tl.where(oc_mask, ex, 0.0)
                s = tl.sum(ex, axis=0)
                sm = ex / s

                max_vals = tl.maximum(max_vals, sm)

    out_base = n * OC * OD * OH * OW + od * OH * OW + oh * OW + ow
    out_offs = out_base + oc_offs * (OD * OH * OW)
    tl.store(out_ptr + out_offs, max_vals, mask=oc_mask)


@triton.jit
def fused_conv_softmax_pool_tile_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    ID, IH, IW,
    CD, CH, CW,
    OD, OH, OW,
    IC: tl.constexpr,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    P: tl.constexpr,
    TILE_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow_tile)
    pid = tl.program_id(0)
    n_ow_tiles = (OW + TILE_W - 1) // TILE_W
    ow_tile = pid % n_ow_tiles
    tmp = pid // n_ow_tiles
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    ow_base = ow_tile * TILE_W

    cd_base = od * P
    ch_base = oh * P
    cw_tile_base = ow_base * P  # base in conv coords

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    x_n_base = n * IC * ID * IH * IW
    KVOL = IC * KD * KH * KW

    # For each ow in tile (static), accumulate max
    for tw in tl.static_range(TILE_W):
        ow_val = ow_base + tw
        cw_base = (ow_base + tw) * P
        max_vals = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)
        if ow_val < OW:
            for di in tl.static_range(P):
                for hi in tl.static_range(P):
                    for wi in tl.static_range(P):
                        cd = cd_base + di
                        ch = ch_base + hi
                        cw = cw_base + wi

                        acc = bias

                        for ic in tl.static_range(IC):
                            for kd in tl.static_range(KD):
                                for kh in tl.static_range(KH):
                                    for kw in tl.static_range(KW):
                                        id_ = cd + kd
                                        ih_ = ch + kh
                                        iw_ = cw + kw
                                        x_off = x_n_base + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                                        xv = tl.load(x_ptr + x_off)
                                        w_off = oc_offs * KVOL + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                                        wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                        acc = acc + xv * wv

                        m = tl.max(tl.where(oc_mask, acc, -float('inf')), axis=0)
                        ex = tl.exp(acc - m)
                        ex = tl.where(oc_mask, ex, 0.0)
                        s = tl.sum(ex, axis=0)
                        sm = ex / s

                        max_vals = tl.maximum(max_vals, sm)

            out_base = n * OC * OD * OH * OW + od * OH * OW + oh * OW + ow_val
            out_offs = out_base + oc_offs * (OD * OH * OW)
            tl.store(out_ptr + out_offs, max_vals, mask=oc_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_factor):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    CD = ID - KD + 1
    CH = IH - KH + 1
    CW = IW - KW + 1
    OD = CD // pool_factor
    OH = CH // pool_factor
    OW = CW // pool_factor

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2
    BLOCK_OC = max(BLOCK_OC, 16)

    grid = (N * OD * OH * OW,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N,
        ID, IH, IW,
        CD, CH, CW,
        OD, OH, OW,
        IC, OC,
        KD, KH, KW,
        pool_factor,
        BLOCK_OC,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_factor = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC, _, KD, KH, KW = weight.shape
        CD = ID - KD + 1
        CH = IH - KH + 1
        CW = IW - KW + 1
        pf = self.pool_factor
        if CD % pf == 0 and CH % pf == 0 and CW % pf == 0:
            return fused_conv_softmax_pool(x, weight, bias, pf)
        else:
            x = self.conv(x)
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x