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
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ids: (n, oc tile, pooled spatial tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    num_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW
    pid_ph = pid_sp // num_pw_tiles
    pid_pw = pid_sp % num_pw_tiles

    # Pooled output coords for this tile
    ph_offs = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]
    ph_mask = ph_offs < POH
    pw_mask = pw_offs < POW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # Conv output rows tile spans BLOCK_PH * POOL rows, BLOCK_PW * POOL cols
    BLOCK_OH: tl.constexpr = BLOCK_PH * POOL
    BLOCK_OW: tl.constexpr = BLOCK_PW * POOL

    oh_offs = pid_ph * BLOCK_OH + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    ow_offs = pid_pw * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    # Flatten conv-output spatial tile to BLOCK_N
    BLOCK_N: tl.constexpr = BLOCK_OH * BLOCK_OW
    sp_idx = tl.arange(0, BLOCK_N)
    oh_local = sp_idx // BLOCK_OW
    ow_local = sp_idx % BLOCK_OW

    oh_flat = pid_ph * BLOCK_OH + oh_local  # [BLOCK_N]
    ow_flat = pid_pw * BLOCK_OW + ow_local  # [BLOCK_N]
    sp_oh_mask = oh_flat < OH
    sp_ow_mask = ow_flat < OW
    sp_mask = sp_oh_mask & sp_ow_mask

    # Accumulator: [BLOCK_OC, BLOCK_N]
    acc = tl.zeros((BLOCK_OC, BLOCK_N), dtype=tl.float32)

    # GEMM-K dimension is IC * KH * KW
    K = IC * KH * KW

    # Loop over K in BLOCK_K chunks
    for k0 in range(0, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_offs < K

        # Decompose k into ic, kh, kw
        ic = k_offs // (KH * KW)
        khkw = k_offs % (KH * KW)
        kh = khkw // KW
        kw = khkw % KW

        # Load weight tile [BLOCK_OC, BLOCK_K]
        # weight layout: [OC, IC, KH, KW] contiguous
        w_idx = oc_offs[:, None] * (IC * KH * KW) + k_offs[None, :]
        w_load_mask = oc_mask[:, None] & k_mask[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=w_load_mask, other=0.0)  # [BLOCK_OC, BLOCK_K]

        # Load input tile [BLOCK_K, BLOCK_N] via implicit im2col
        ih = oh_flat[None, :] + kh[:, None]  # [BLOCK_K, BLOCK_N]
        iw = ow_flat[None, :] + kw[:, None]  # [BLOCK_K, BLOCK_N]
        # Clamp to valid range to avoid OOB reads
        ih_safe = tl.where(sp_mask[None, :], ih, 0)
        iw_safe = tl.where(sp_mask[None, :], iw, 0)
        ic_safe = tl.where(k_mask[:, None], ic[:, None], 0)
        x_idx = (pid_n.to(tl.int64) * IC * IH * IW) + ic_safe * (IH * IW) + ih_safe * IW + iw_safe
        x_load_mask = k_mask[:, None] & sp_mask[None, :]
        x_tile = tl.load(x_ptr + x_idx, mask=x_load_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc += tl.dot(w_tile, x_tile)

    # Add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bias[:, None]

    # Apply tanh(x - SUB1) - SUB2
    v = acc - SUB1
    t = 2.0 * tl.sigmoid(2.0 * v) - 1.0
    t = t - SUB2  # [BLOCK_OC, BLOCK_N]

    # Reshape to [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL] and reduce POOL dims
    # We do this by indexing.
    # t is laid out as [BLOCK_OC, BLOCK_OH * BLOCK_OW]
    # We want pooled[BLOCK_OC, BLOCK_PH, BLOCK_PW] = mean over POOLxPOOL window

    pool_inv = 1.0 / (POOL * POOL)

    # Pooled accumulator
    pool_acc = tl.zeros((BLOCK_OC, BLOCK_PH * BLOCK_PW), dtype=tl.float32)

    # Build pool mapping: for each pooled position (ph_l, pw_l), sum over (dh, dw)
    pidx = tl.arange(0, BLOCK_PH * BLOCK_PW)
    ph_l = pidx // BLOCK_PW  # [BLOCK_PH*BLOCK_PW]
    pw_l = pidx % BLOCK_PW

    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            # corresponding flat index in t
            oh_l = ph_l * POOL + dh
            ow_l = pw_l * POOL + dw
            flat = oh_l * BLOCK_OW + ow_l  # [BLOCK_PH*BLOCK_PW]
            # gather: t[:, flat]
            # Use tl.load on a temp? Actually we have t in registers.
            # We need to gather from t along axis 1. Use multiplication by one-hot? 
            # Simpler: compute the indices and use tl.gather - but Triton may not support arbitrary gather.
            # Instead, restructure: iterate dh,dw over conv output, and for each (dh,dw) only sum the matching positions.
            # We use a mask-based reduction.
            pass

    # Alternative approach: use the layout directly.
    # t has shape [BLOCK_OC, BLOCK_OH*BLOCK_OW]. We pool by reshaping conceptually.
    # Triton supports tl.reshape on tiles when dims are constexpr.
    t4 = tl.reshape(t, (BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL))
    # Sum over the two POOL axes
    t_sum = tl.sum(t4, axis=4)  # [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW]
    t_sum = tl.sum(t_sum, axis=2)  # [BLOCK_OC, BLOCK_PH, BLOCK_PW]
    pooled = t_sum * pool_inv  # [BLOCK_OC, BLOCK_PH, BLOCK_PW]

    # Store
    # out layout: [N, OC, POH, POW]
    # offset = n*OC*POH*POW + oc*POH*POW + ph*POW + pw
    out_base = pid_n * OC * POH * POW
    out_idx = (
        out_base
        + oc_offs[:, None, None] * (POH * POW)
        + ph_offs[None, :, None] * POW
        + pw_offs[None, None, :]
    )
    out_mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    tl.store(out_ptr + out_idx, pooled, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = float(subtract1_value)
        self.subtract2_value = float(subtract2_value)
        self.kernel_size_pool = kernel_size_pool
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

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

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 128
        BLOCK_PH = 4
        BLOCK_PW = 16  # -> BLOCK_OH=8, BLOCK_OW=32, BLOCK_N=256
        BLOCK_K = 64

        num_pw_tiles = (POW + BLOCK_PW - 1) // BLOCK_PW
        num_ph_tiles = (POH + BLOCK_PH - 1) // BLOCK_PH

        grid = (N, triton.cdiv(OC, BLOCK_OC), num_ph_tiles * num_pw_tiles)

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            POH, POW,
            KH, KW,
            POOL,
            self.subtract1_value, self.subtract2_value,
            BLOCK_OC, BLOCK_PH, BLOCK_PW, BLOCK_K,
            num_warps=8, num_stages=3,
        )
        return out