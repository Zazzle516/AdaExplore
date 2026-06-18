import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PH': 8, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_PH': 8, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=3),
    ],
    key=['IH', 'IW', 'IC', 'OC'],
)
@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC,
    OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
    IC_VAL: tl.constexpr,
    KH_VAL: tl.constexpr,
    KW_VAL: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    grid_pw = tl.cdiv(PW, BLOCK_PW)
    pid_ph = tl.program_id(2) // grid_pw
    pid_pw = tl.program_id(2) % grid_pw

    TILE_OH: tl.constexpr = BLOCK_PH * POOL
    TILE_OW: tl.constexpr = BLOCK_PW * POOL

    # conv output coordinates this program computes
    oh_base = pid_ph * TILE_OH
    ow_base = pid_pw * TILE_OW
    oh_offs = oh_base + tl.arange(0, TILE_OH)  # [TILE_OH]
    ow_offs = ow_base + tl.arange(0, TILE_OW)  # [TILE_OW]
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW
    out_mask = oh_mask[:, None] & ow_mask[None, :]

    bias = tl.load(b_ptr + pid_oc)
    conv_tile = tl.zeros((TILE_OH, TILE_OW), dtype=tl.float32)

    for ic in tl.static_range(IC_VAL):
        for kh in tl.static_range(KH_VAL):
            for kw in tl.static_range(KW_VAL):
                w_off = ((pid_oc * IC_VAL + ic) * KH_VAL + kh) * KW_VAL + kw
                w_val = tl.load(w_ptr + w_off)
                ih = oh_offs + kh
                iw = ow_offs + kw
                x_off = ((pid_n * IC_VAL + ic) * IH + ih[:, None]) * IW + iw[None, :]
                x_val = tl.load(x_ptr + x_off, mask=out_mask, other=0.0)
                conv_tile += x_val * w_val

    conv_tile += bias

    # average pool POOL x POOL: reshape to (BLOCK_PH, POOL, BLOCK_PW, POOL) and sum
    reshaped = tl.reshape(conv_tile, (BLOCK_PH, POOL, BLOCK_PW, POOL))
    pooled = tl.sum(reshaped, axis=3)
    pooled = tl.sum(pooled, axis=1)
    pooled = pooled / (POOL * POOL)

    sig = 1.0 / (1.0 + tl.exp(-pooled))
    ph_offs = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)
    pmask = (ph_offs[:, None] < PH) & (pw_offs[None, :] < PW)
    sig = tl.where(pmask, sig, 0.0)
    partial = tl.sum(tl.sum(sig, axis=1), axis=0)

    tl.atomic_add(out_ptr + pid_n, partial)


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

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        def grid(meta):
            grid_pw = (PW + meta['BLOCK_PW'] - 1) // meta['BLOCK_PW']
            grid_ph = (PH + meta['BLOCK_PH'] - 1) // meta['BLOCK_PH']
            return (N, OC, grid_ph * grid_pw)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC,
            OH, OW,
            PH, PW,
            POOL=POOL,
            IC_VAL=IC,
            KH_VAL=KH,
            KW_VAL=KW,
        )

        return out