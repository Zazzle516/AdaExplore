import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_OH': 2, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 2, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def conv_tanh_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    SUB1: tl.constexpr, SUB2: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    """
    Each program computes a BLOCK_OH x BLOCK_OW tile of conv output for BLOCK_OC channels,
    applies bias - SUB1, tanh, - SUB2, then pools over POOL x POOL windows.
    BLOCK_OH and BLOCK_OW are required to be multiples of POOL.
    """
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    # tile grid in output space
    n_tiles_w = tl.cdiv(OW, BLOCK_OW)
    pid_oh = pid_sp // n_tiles_w
    pid_ow = pid_sp % n_tiles_w

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)        # [BLOCK_OC]
    oh_offs = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)        # [BLOCK_OH]
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)        # [BLOCK_OW]

    oc_mask = oc_offs < OC
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)    # [BLOCK_OC]

    # accumulator for conv output: [BLOCK_OC, BLOCK_OH * BLOCK_OW]
    HW = BLOCK_OH * BLOCK_OW
    acc = tl.zeros((BLOCK_OC, HW), dtype=tl.float32)

    # flatten output spatial coordinates
    flat_idx = tl.arange(0, HW)
    fl_oh = flat_idx // BLOCK_OW    # [HW]
    fl_ow = flat_idx % BLOCK_OW     # [HW]

    # input row/col for each output spatial point (base; will offset by kh/kw)
    base_ih = (pid_oh * BLOCK_OH + fl_oh)   # [HW]
    base_iw = (pid_ow * BLOCK_OW + fl_ow)   # [HW]
    sp_mask = (base_ih < OH) & (base_iw < OW)

    # Loop over IC (outer), then (kh,kw) static
    for ic in range(0, IC):
        for kh in tl.static_range(KH):
            for kw in tl.static_range(KW):
                ih = base_ih + kh    # [HW]
                iw = base_iw + kw    # [HW]
                x_off = ((pid_n * IC + ic) * IH + ih) * IW + iw
                x_mask_v = sp_mask & (ih < IH) & (iw < IW)
                x_val = tl.load(x_ptr + x_off, mask=x_mask_v, other=0.0)  # [HW]

                w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)   # [BLOCK_OC]

                acc += w_val[:, None] * x_val[None, :]

    # Bias + sub1 + tanh + sub2
    v = acc + bias[:, None] - SUB1
    t = 2.0 * tl.sigmoid(2.0 * v) - 1.0 - SUB2

    # Reshape to (BLOCK_OC, BLOCK_OH, BLOCK_OW) and pool
    t3 = tl.reshape(t, (BLOCK_OC, BLOCK_OH, BLOCK_OW))

    # We need to pool POOL x POOL windows. Assume BLOCK_OH % POOL == 0 and BLOCK_OW % POOL == 0.
    PH_TILE: tl.constexpr = BLOCK_OH // POOL
    PW_TILE: tl.constexpr = BLOCK_OW // POOL

    # Reshape: (BLOCK_OC, PH_TILE, POOL, PW_TILE, POOL)
    t5 = tl.reshape(t3, (BLOCK_OC, PH_TILE, POOL, PW_TILE, POOL))
    # Sum over POOL dims
    s = tl.sum(t5, axis=4)   # (BLOCK_OC, PH_TILE, POOL, PW_TILE)
    s = tl.sum(s, axis=2)    # (BLOCK_OC, PH_TILE, PW_TILE)

    inv = 1.0 / (POOL * POOL)
    pooled = s * inv

    # output coords
    ph_offs = pid_oh * (BLOCK_OH // POOL) + tl.arange(0, PH_TILE)   # [PH_TILE]
    pw_offs = pid_ow * (BLOCK_OW // POOL) + tl.arange(0, PW_TILE)   # [PW_TILE]
    ph_mask = ph_offs < PH
    pw_mask = pw_offs < PW

    # Store: flatten pooled (BLOCK_OC, PH_TILE * PW_TILE)
    pooled_flat = tl.reshape(pooled, (BLOCK_OC, PH_TILE * PW_TILE))
    p_flat_idx = tl.arange(0, PH_TILE * PW_TILE)
    p_h = p_flat_idx // PW_TILE
    p_w = p_flat_idx % PW_TILE
    ph_full = pid_oh * (BLOCK_OH // POOL) + p_h
    pw_full = pid_ow * (BLOCK_OW // POOL) + p_w
    out_off = ((pid_n * OC + oc_offs[:, None]) * PH + ph_full[None, :]) * PW + pw_full[None, :]
    out_mask = oc_mask[:, None] & (ph_full[None, :] < PH) & (pw_full[None, :] < PW)
    tl.store(out_ptr + out_off, pooled_flat, mask=out_mask)


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
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.kernel_size_pool
        PH = OH // POOL
        PW = OW // POOL

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)

        def grid(META):
            n_tiles_h = triton.cdiv(OH, META['BLOCK_OH'])
            n_tiles_w = triton.cdiv(OW, META['BLOCK_OW'])
            return (N, triton.cdiv(OC, META['BLOCK_OC']), n_tiles_h * n_tiles_w)

        conv_tanh_pool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            self.subtract1_value, self.subtract2_value,
            KH, KW, POOL,
        )
        return out