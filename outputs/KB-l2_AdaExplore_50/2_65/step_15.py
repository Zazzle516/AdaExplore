import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC,
    PH, PW,
    POOL: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
    IC_VAL: tl.constexpr,
    KH_VAL: tl.constexpr,
    KW_VAL: tl.constexpr,
):
    # one program per (n, oc, pool-tile-h, pool-tile-w)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_ph = tl.program_id(2) // ((PW + BLOCK_PW - 1) // BLOCK_PW)
    pid_pw = tl.program_id(2) % ((PW + BLOCK_PW - 1) // BLOCK_PW)

    ph_offs = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_offs = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]
    ph_mask = ph_offs < PH
    pw_mask = pw_offs < PW
    mask2d = ph_mask[:, None] & pw_mask[None, :]  # [BLOCK_PH, BLOCK_PW]

    bias = tl.load(b_ptr + pid_oc)

    # Tile region of input needed:
    # input rows: ph_base*POOL .. ph_base*POOL + (BLOCK_PH*POOL + KH_VAL - 1)
    # similar for cols. We compute per-output conv vals on the fly.

    # Accumulator over conv outputs in this pool tile (sum over POOL*POOL window)
    acc = tl.zeros((BLOCK_PH, BLOCK_PW), dtype=tl.float32)

    base_h = ph_offs * POOL  # [BLOCK_PH] - top of pool window in conv output coords
    base_w = pw_offs * POOL  # [BLOCK_PW]

    # Loop over pool window (dh, dw)
    for dh in tl.static_range(POOL):
        for dw in tl.static_range(POOL):
            # conv output coord (oh, ow) = (base_h+dh, base_w+dw)
            # which equals input top-left (oh, ow), and input pos is (oh+kh, ow+kw)
            conv_val = tl.zeros((BLOCK_PH, BLOCK_PW), dtype=tl.float32)
            for ic in tl.static_range(IC_VAL):
                for kh in tl.static_range(KH_VAL):
                    for kw in tl.static_range(KW_VAL):
                        ih = base_h + dh + kh  # [BLOCK_PH]
                        iw = base_w + dw + kw  # [BLOCK_PW]
                        x_off = ((pid_n * IC + ic) * IH + ih[:, None]) * IW + iw[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=mask2d, other=0.0)
                        w_off = ((pid_oc * IC + ic) * KH_VAL + kh) * KW_VAL + kw
                        w_val = tl.load(w_ptr + w_off)
                        conv_val += x_val * w_val
            conv_val += bias
            acc += conv_val

    pooled = acc / (POOL * POOL)
    sig = 1.0 / (1.0 + tl.exp(-pooled))
    sig = tl.where(mask2d, sig, 0.0)
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

        BLOCK_PH = 8
        BLOCK_PW = 16
        grid_pw = (PW + BLOCK_PW - 1) // BLOCK_PW
        grid_ph = (PH + BLOCK_PH - 1) // BLOCK_PH
        grid = (N, OC, grid_ph * grid_pw)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC,
            PH, PW,
            POOL=POOL,
            BLOCK_PH=BLOCK_PH,
            BLOCK_PW=BLOCK_PW,
            IC_VAL=IC,
            KH_VAL=KH,
            KW_VAL=KW,
            num_warps=4,
            num_stages=2,
        )

        return out