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
    OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    # one program per (n, oc-tile, pool-tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)         # [BLOCK_P]
    p_mask = p_offs < (PH * PW)
    ph = p_offs // PW
    pw = p_offs % PW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)     # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # Load bias: [BLOCK_OC]
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    inv_pool2 = 1.0 / (POOL * POOL)

    # accumulator over the BLOCK_OC x BLOCK_P pooled values
    pooled_acc = tl.zeros([BLOCK_OC, BLOCK_P], dtype=tl.float32)

    for dy in tl.static_range(0, POOL):
        for dx in tl.static_range(0, POOL):
            oh = ph * POOL + dy        # [BLOCK_P]
            ow = pw * POOL + dx        # [BLOCK_P]

            conv_val = tl.zeros([BLOCK_OC, BLOCK_P], dtype=tl.float32) + bias[:, None]

            for ic in tl.static_range(0, IC_C):
                for kh in tl.static_range(0, KH_C):
                    for kw in tl.static_range(0, KW_C):
                        ih = oh + kh    # [BLOCK_P]
                        iw = ow + kw    # [BLOCK_P]
                        x_off = ((pid_n * IC + ic) * H + ih) * W + iw  # [BLOCK_P]
                        x_m = p_mask & (ih < H) & (iw < W)
                        xv = tl.load(x_ptr + x_off, mask=x_m, other=0.0)  # [BLOCK_P]

                        # w[oc, ic, kh, kw]: [BLOCK_OC]
                        w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw
                        wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                        conv_val += wv[:, None] * xv[None, :]

            pooled_acc += conv_val * inv_pool2

    sig = tl.sigmoid(pooled_acc)
    # mask invalid positions
    full_mask = oc_mask[:, None] & p_mask[None, :]
    sig = tl.where(full_mask, sig, 0.0)
    s = tl.sum(tl.sum(sig, axis=0), axis=0)

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

        BLOCK_P = 64
        BLOCK_OC = 16
        n_p_tiles = (PH * PW + BLOCK_P - 1) // BLOCK_P
        n_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (N, n_oc_tiles, n_p_tiles)
        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW, PH, PW,
            POOL=POOL,
            BLOCK_P=BLOCK_P,
            BLOCK_OC=BLOCK_OC,
            KH_C=KH,
            KW_C=KW,
            IC_C=IC,
            num_warps=4,
            num_stages=2,
        )
        return out