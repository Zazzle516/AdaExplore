import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    BLOCK_PW: tl.constexpr,
    POOL: tl.constexpr,
    IC_VAL: tl.constexpr,
    KH_VAL: tl.constexpr,
    KW_VAL: tl.constexpr,
    OC_TILE: tl.constexpr,
):
    # Grid: (N, OC // OC_TILE, PH * ceil(PW/BLOCK_PW))
    pid_n = tl.program_id(0)
    pid_oc_tile = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
    ph = pid_p // num_pw_tiles
    pw_tile = pid_p % num_pw_tiles

    pw_offs = pw_tile * BLOCK_PW + tl.arange(0, BLOCK_PW)
    pw_mask = pw_offs < PW

    base_oh = ph * POOL
    ow_base = pw_tile * BLOCK_PW * POOL
    NWCOLS: tl.constexpr = BLOCK_PW * POOL
    ow_offs = ow_base + tl.arange(0, NWCOLS)
    ow_mask = ow_offs < OW

    oc_base = pid_oc_tile * OC_TILE

    # Accumulate the final scalar partial sum for this program over OC_TILE channels
    program_partial = tl.zeros((), dtype=tl.float32)

    # Loop over output channels in this tile
    for oc_inner in tl.static_range(OC_TILE):
        oc_idx = oc_base + oc_inner
        bias = tl.load(b_ptr + oc_idx)

        pool_acc = tl.zeros((BLOCK_PW, POOL), dtype=tl.float32)

        for kh_idx in tl.static_range(POOL):
            oh = base_oh + kh_idx
            conv_row = tl.zeros((NWCOLS,), dtype=tl.float32)
            for ic in tl.static_range(IC_VAL):
                for kh in tl.static_range(KH_VAL):
                    ih = oh + kh
                    for kw in tl.static_range(KW_VAL):
                        iw = ow_offs + kw
                        x_off = ((pid_n * IC_VAL + ic) * IH + ih) * IW + iw
                        x_val = tl.load(x_ptr + x_off, mask=ow_mask, other=0.0)
                        w_off = ((oc_idx * IC_VAL + ic) * KH_VAL + kh) * KW_VAL + kw
                        w_val = tl.load(w_ptr + w_off)
                        conv_row += x_val * w_val
            conv_row += bias
            conv_row_2d = tl.reshape(conv_row, (BLOCK_PW, POOL))
            pool_acc += conv_row_2d

        pooled = pool_acc / (POOL * POOL)
        pooled_sum = tl.sum(pooled, axis=1)
        sig = 1.0 / (1.0 + tl.exp(-pooled_sum))
        sig = tl.where(pw_mask, sig, 0.0)
        partial = tl.sum(sig, axis=0)
        program_partial += partial

    tl.atomic_add(out_ptr + pid_n, program_partial)


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

        BLOCK_PW = 32
        OC_TILE = 4
        assert OC % OC_TILE == 0
        num_pw_tiles = (PW + BLOCK_PW - 1) // BLOCK_PW
        grid = (N, OC // OC_TILE, PH * num_pw_tiles)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            BLOCK_PW=BLOCK_PW,
            POOL=POOL,
            IC_VAL=IC,
            KH_VAL=KH,
            KW_VAL=KW,
            OC_TILE=OC_TILE,
            num_warps=4,
            num_stages=2,
        )

        return out