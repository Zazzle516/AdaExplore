import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,           # conv output dims
    PH, PW,           # pooled dims
    POOL: tl.constexpr,
    BLOCK_PW: tl.constexpr,   # pooled-cols per program
    BLOCK_PH: tl.constexpr,   # pooled-rows per program
    OC_C: tl.constexpr,       # = OC (compile-time)
    IC_C: tl.constexpr,       # = IC
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    # one program per (n, ph_tile, pw_tile)
    pid_n = tl.program_id(0)
    pid_ph = tl.program_id(1)
    pid_pw = tl.program_id(2)

    ph_base = pid_ph * BLOCK_PH
    pw_base = pid_pw * BLOCK_PW

    # conv-output region this program covers
    # rows: ph_base*POOL .. ph_base*POOL + BLOCK_PH*POOL - 1
    # cols: pw_base*POOL .. pw_base*POOL + BLOCK_PW*POOL - 1

    OH_TILE: tl.constexpr = BLOCK_PH * POOL
    OW_TILE: tl.constexpr = BLOCK_PW * POOL

    # input footprint size
    IH_TILE: tl.constexpr = OH_TILE + KH_C - 1
    IW_TILE: tl.constexpr = OW_TILE + KW_C - 1

    oc_range = tl.arange(0, OC_C)
    bias = tl.load(b_ptr + oc_range)  # [OC_C]

    inv_pool2 = 1.0 / (POOL * POOL)

    # accumulator over OC for this (n) — pooled, sigmoid, summed over OC and pool tile
    final_acc = tl.zeros([1], dtype=tl.float32)

    # We'll accumulate sigmoid sums in a scalar.
    # Strategy: produce conv outputs for the OH_TILE x OW_TILE block, one (kh,kw,ic) at a time,
    # accumulating into a [OH_TILE*OW_TILE, OC_C] register block.

    # Output conv accumulator: [OH_TILE*OW_TILE, OC_C]
    M = OH_TILE * OW_TILE
    conv_acc = tl.zeros([OH_TILE * OW_TILE, OC_C], dtype=tl.float32)

    # add bias
    conv_acc += bias[None, :]

    # output row/col indices within tile
    out_idx = tl.arange(0, OH_TILE * OW_TILE)
    out_r = out_idx // OW_TILE  # [M]
    out_c = out_idx % OW_TILE   # [M]

    oh_global = ph_base * POOL + out_r
    ow_global = pw_base * POOL + out_c

    # For each (ic, kh, kw) we accumulate conv_acc[m, oc] += x[m] * w[oc]
    # x[m] = input[n, ic, oh_global[m]+kh, ow_global[m]+kw]
    # w[oc] = weight[oc, ic, kh, kw]

    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH_C):
            for kw in tl.static_range(0, KW_C):
                ih = oh_global + kh
                iw = ow_global + kw
                x_off = ((pid_n * IC + ic) * H + ih) * W + iw
                x_mask = (ih < H) & (iw < W)
                xv = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [M]
                w_off = ((oc_range * IC + ic) * KH + kh) * KW + kw
                wv = tl.load(w_ptr + w_off)  # [OC_C]
                conv_acc += xv[:, None] * wv[None, :]

    # Now we have conv outputs for the OH_TILE x OW_TILE block across all OC.
    # Apply average pool over POOLxPOOL windows -> [BLOCK_PH*BLOCK_PW, OC_C]
    # Then sigmoid, then sum over both axes.

    # Reshape mentally: out_r // POOL = ph_local, out_r % POOL = dy
    #                   out_c // POOL = pw_local, out_c % POOL = dx
    # pool cell index = ph_local * BLOCK_PW + pw_local
    ph_local = out_r // POOL
    pw_local = out_c // POOL
    pool_idx = ph_local * BLOCK_PW + pw_local  # [M], values in [0, BLOCK_PH*BLOCK_PW)

    # Sum conv_acc entries with same pool_idx -> pooled[BLOCK_PH*BLOCK_PW, OC_C]
    # Use tl.zeros and atomic-like reduction via mask sum:
    NP: tl.constexpr = BLOCK_PH * BLOCK_PW
    pooled = tl.zeros([NP, OC_C], dtype=tl.float32)

    # For each pool cell, sum the POOL*POOL entries.
    # We do this by iterating dy, dx (static) and gathering rows.
    for dy in tl.static_range(0, POOL):
        for dx in tl.static_range(0, POOL):
            # For pool cell p (with ph_local, pw_local), the conv row index =
            #   (ph_local*POOL + dy) * OW_TILE + (pw_local*POOL + dx)
            p_idx = tl.arange(0, NP)
            ph_l = p_idx // BLOCK_PW
            pw_l = p_idx % BLOCK_PW
            r = ph_l * POOL + dy
            c = pw_l * POOL + dx
            row = r * OW_TILE + c  # [NP]
            # gather conv_acc[row, :]
            # We need to load from conv_acc; but conv_acc is a register tile, can't gather.
            # Instead use masked-sum approach: pooled += where(pool_idx_matches, conv_acc, 0)
            # We'll use a different approach below.
            pass

    # Use direct broadcast/where reduction:
    # Build per-pool mask [NP, M] would be too large. Instead, use the fact that
    # M = NP * POOL * POOL with a known permutation. We can reshape conv_acc.
    # conv_acc layout: row m = out_r * OW_TILE + out_c
    # We want to reindex to [ph_local, dy, pw_local, dx, OC] then sum over (dy, dx).
    # Since M and NP are constexpr, we can use tl.reshape.

    # reshape conv_acc [M, OC_C] -> [BLOCK_PH, POOL, BLOCK_PW, POOL, OC_C]
    conv_5d = tl.reshape(conv_acc, [BLOCK_PH, POOL, BLOCK_PW, POOL, OC_C])
    # sum over dy (axis=1) and dx (axis=3)
    pooled_sum = tl.sum(conv_5d, axis=3)   # [BLOCK_PH, POOL, BLOCK_PW, OC_C]
    pooled_sum = tl.sum(pooled_sum, axis=1)  # [BLOCK_PH, BLOCK_PW, OC_C]

    pooled_vals = pooled_sum * inv_pool2

    # Mask invalid pooled positions
    p_idx = tl.arange(0, BLOCK_PH)
    q_idx = tl.arange(0, BLOCK_PW)
    ph_global = ph_base + p_idx  # [BLOCK_PH]
    pw_global = pw_base + q_idx  # [BLOCK_PW]
    valid = (ph_global[:, None] < PH) & (pw_global[None, :] < PW)  # [BLOCK_PH, BLOCK_PW]

    sig = tl.sigmoid(pooled_vals)  # [BLOCK_PH, BLOCK_PW, OC_C]
    sig = tl.where(valid[:, :, None], sig, 0.0)
    s = tl.sum(tl.sum(tl.sum(sig, axis=2), axis=1), axis=0)

    tl.atomic_add(out_ptr + pid_n, s)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        POOL = self.pool_kernel_size

        OH = H - KH + 1
        OW = W - KW + 1
        PH = OH // POOL
        PW = OW // POOL

        if PH * POOL != OH or PW * POOL != OW:
            y = self.conv(x)
            y = self.avg_pool(y)
            y = torch.sigmoid(y)
            return torch.sum(y, dim=[1, 2, 3])

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        BLOCK_PH = 2
        BLOCK_PW = 8
        n_ph = (PH + BLOCK_PH - 1) // BLOCK_PH
        n_pw = (PW + BLOCK_PW - 1) // BLOCK_PW

        grid = (N, n_ph, n_pw)
        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW, PH, PW,
            POOL=POOL,
            BLOCK_PW=BLOCK_PW,
            BLOCK_PH=BLOCK_PH,
            OC_C=OC,
            IC_C=IC,
            KH_C=KH,
            KW_C=KW,
            num_warps=8,
            num_stages=2,
        )
        return out