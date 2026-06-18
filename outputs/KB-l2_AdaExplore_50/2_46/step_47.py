import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC,
    OH, OW,        # conv output spatial
    POH, POW,      # pooled output spatial
    sub1, sub2,
    BLOCK_OC: tl.constexpr,
    BLOCK_POH: tl.constexpr,
    BLOCK_POW: tl.constexpr,
    PK: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    NUM_POW_TILES = (POW + BLOCK_POW - 1) // BLOCK_POW
    pid_poh = pid_sp // NUM_POW_TILES
    pid_pow = pid_sp % NUM_POW_TILES

    BLOCK_OH: tl.constexpr = BLOCK_POH * PK
    BLOCK_OW: tl.constexpr = BLOCK_POW * PK

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)        # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # conv output spatial offsets for this tile
    oh_offs = pid_poh * BLOCK_OH + tl.arange(0, BLOCK_OH)       # [BLOCK_OH]
    ow_offs = pid_pow * BLOCK_OW + tl.arange(0, BLOCK_OW)       # [BLOCK_OW]
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    # flatten spatial into [BLOCK_OH*BLOCK_OW]
    sp_oh = (oh_offs[:, None] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32))
    sp_ow = (ow_offs[None, :] + tl.zeros((BLOCK_OH, BLOCK_OW), dtype=tl.int32))
    sp_mask = oh_mask[:, None] & ow_mask[None, :]               # [BLOCK_OH, BLOCK_OW]

    # conv accumulator: [BLOCK_OC, BLOCK_OH*BLOCK_OW]
    BLOCK_SP: tl.constexpr = BLOCK_OH * BLOCK_OW
    conv_acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    sp_oh_flat = tl.reshape(sp_oh, (BLOCK_SP,))
    sp_ow_flat = tl.reshape(sp_ow, (BLOCK_SP,))
    sp_mask_flat = tl.reshape(sp_mask, (BLOCK_SP,))

    x_batch_off = pid_n * IC * H * W

    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = sp_oh_flat + kh
                iw = sp_ow_flat + kw
                in_off = x_batch_off + ic * H * W + ih * W + iw
                in_mask = sp_mask_flat & (ih < H) & (iw < W)
                x_val = tl.load(x_ptr + in_off, mask=in_mask, other=0.0)  # [BLOCK_SP]

                w_off = oc_offs * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)   # [BLOCK_OC]

                conv_acc += w_val[:, None] * x_val[None, :]

    # bias + sub1 + tanh + sub2
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    v = conv_acc + b_val[:, None] - sub1
    e2 = tl.exp(2.0 * v)
    t = (e2 - 1.0) / (e2 + 1.0)
    v = t - sub2  # [BLOCK_OC, BLOCK_SP]

    # reshape to [BLOCK_OC, BLOCK_OH, BLOCK_OW] then pool
    v_2d = tl.reshape(v, (BLOCK_OC, BLOCK_OH, BLOCK_OW))

    # pool via static range
    inv = 1.0 / (PK * PK)
    pool_acc = tl.zeros((BLOCK_OC, BLOCK_POH, BLOCK_POW), dtype=tl.float32)
    for pi in tl.static_range(0, PK):
        for pj in tl.static_range(0, PK):
            # slice v_2d[:, pi::PK, pj::PK] equivalent
            # build indices
            poh_r = tl.arange(0, BLOCK_POH) * PK + pi   # [BLOCK_POH]
            pow_r = tl.arange(0, BLOCK_POW) * PK + pj   # [BLOCK_POW]
            # gather using arithmetic: flatten index
            idx = poh_r[:, None] * BLOCK_OW + pow_r[None, :]   # [BLOCK_POH, BLOCK_POW]
            idx_flat = tl.reshape(idx, (BLOCK_POH * BLOCK_POW,))
            # gather from v: [BLOCK_OC, BLOCK_SP] -> use take by indexing
            # we use broadcasted load via tl.gather-like: shift pointers? Use simple approach:
            # since v is a register tile, just slice with .reshape and stride trick
            # Implement gather via tl.where mask sum? Easier: rebuild from v_2d by manual slicing using arange
            pass
    # Fall back to direct manual sum via per-(pi,pj) reshape trick:
    pool_acc = tl.zeros((BLOCK_OC, BLOCK_POH * BLOCK_POW), dtype=tl.float32)
    for pi in tl.static_range(0, PK):
        for pj in tl.static_range(0, PK):
            poh_r = tl.arange(0, BLOCK_POH) * PK + pi
            pow_r = tl.arange(0, BLOCK_POW) * PK + pj
            idx2d = poh_r[:, None] * BLOCK_OW + pow_r[None, :]
            idx_flat = tl.reshape(idx2d, (BLOCK_POH * BLOCK_POW,))
            # gather: use tl.load on a dummy? No — v is in registers.
            # Use mask trick: compute per-(pi,pj) by re-reading from conv? expensive.
            # Instead: use sum over masked v where mask selects positions matching (pi,pj).
            sp_oh_idx = tl.arange(0, BLOCK_OH)
            sp_ow_idx = tl.arange(0, BLOCK_OW)
            m_oh = (sp_oh_idx % PK) == pi   # [BLOCK_OH]
            m_ow = (sp_ow_idx % PK) == pj   # [BLOCK_OW]
            m_2d = m_oh[:, None] & m_ow[None, :]
            m_flat = tl.reshape(m_2d, (BLOCK_SP,))
            # masked v: zero where not matching
            v_masked = tl.where(m_flat[None, :], v, 0.0)
            # reshape into [BLOCK_OC, BLOCK_POH, PK, BLOCK_POW, PK] and sum on PK dims? simpler:
            # reshape [BLOCK_OC, BLOCK_OH, BLOCK_OW] -> sum over pi,pj positions
            v_m_2d = tl.reshape(v_masked, (BLOCK_OC, BLOCK_OH, BLOCK_OW))
            # downsample by summing PK-blocks: reshape to [BLOCK_OC, BLOCK_POH, PK, BLOCK_POW, PK]
            v_m_5d = tl.reshape(v_m_2d, (BLOCK_OC, BLOCK_POH, PK, BLOCK_POW, PK))
            v_sum = tl.sum(tl.sum(v_m_5d, axis=4), axis=2)  # [BLOCK_OC, BLOCK_POH, BLOCK_POW]
            pool_acc += tl.reshape(v_sum, (BLOCK_OC, BLOCK_POH * BLOCK_POW))

    pool_acc = pool_acc * inv

    # store: output shape [N, OC, POH, POW]
    poh_offs = pid_poh * BLOCK_POH + tl.arange(0, BLOCK_POH)
    pow_offs = pid_pow * BLOCK_POW + tl.arange(0, BLOCK_POW)
    poh_m = poh_offs < POH
    pow_m = pow_offs < POW

    out_sp_off = poh_offs[:, None] * POW + pow_offs[None, :]            # [BLOCK_POH, BLOCK_POW]
    out_sp_mask = poh_m[:, None] & pow_m[None, :]
    out_sp_off_flat = tl.reshape(out_sp_off, (BLOCK_POH * BLOCK_POW,))
    out_sp_mask_flat = tl.reshape(out_sp_mask, (BLOCK_POH * BLOCK_POW,))

    out_off = (pid_n * OC * POH * POW
               + oc_offs[:, None] * (POH * POW)
               + out_sp_off_flat[None, :])
    out_mask = oc_mask[:, None] & out_sp_mask_flat[None, :]
    tl.store(out_ptr + out_off, pool_acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract1_value, subtract2_value, kernel_size_pool):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract1_value = subtract1_value
        self.subtract2_value = subtract2_value
        self.avgpool = nn.AvgPool2d(kernel_size_pool)
        self.kernel_size = kernel_size
        self.kernel_size_pool = kernel_size_pool
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        PK = self.kernel_size_pool
        POH = OH // PK
        POW = OW // PK

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_POH = 4
        BLOCK_POW = 4

        num_poh_tiles = (POH + BLOCK_POH - 1) // BLOCK_POH
        num_pow_tiles = (POW + BLOCK_POW - 1) // BLOCK_POW

        grid = (
            N,
            (OC + BLOCK_OC - 1) // BLOCK_OC,
            num_poh_tiles * num_pow_tiles,
        )

        fused_conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC,
            OH, OW,
            POH, POW,
            float(self.subtract1_value), float(self.subtract2_value),
            BLOCK_OC=BLOCK_OC,
            BLOCK_POH=BLOCK_POH,
            BLOCK_POW=BLOCK_POW,
            PK=PK,
            KH=KH,
            KW=KW,
            num_warps=8,
            num_stages=3,
        )
        return out