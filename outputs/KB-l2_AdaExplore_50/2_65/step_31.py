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
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < (PH * PW)
    ph = p_offs // PW
    pw = p_offs % PW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    inv_pool2 = 1.0 / (POOL * POOL)

    # base output indices for pool window (top-left of each pool block)
    base_oh = ph * POOL  # [BLOCK_P]
    base_ow = pw * POOL  # [BLOCK_P]

    # Pre-load full weight tile [BLOCK_OC, IC*KH*KW] into registers
    # Loop over (kh, kw, dy, dx) and inner ic accumulating with shared loads.

    # accumulator over the pooled value, summed across pool window
    pool_sum = tl.zeros([BLOCK_OC, BLOCK_P], dtype=tl.float32) + bias[:, None] * (POOL * POOL)

    # For each (dy, dx) in pool window, compute conv output and accumulate (sum, no bias here)
    # We unrolled bias above by multiplying by pool^2 (since we add bias for each pool cell).
    # So in the inner loops we only add ic/kh/kw contributions, summed over all dy,dx.

    # Loop order: ic, kh, kw outer; for each weight, we accumulate over (dy, dx) inner
    # That way weight is loaded once per (oc, ic, kh, kw).
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH_C):
            for kw in tl.static_range(0, KW_C):
                w_off = ((oc_offs * IC + ic) * KH + kh) * KW + kw  # [BLOCK_OC]
                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                # x_sum[p] = sum_{dy,dx} x[n, ic, base_oh+dy+kh, base_ow+dx+kw]
                x_acc = tl.zeros([BLOCK_P], dtype=tl.float32)
                for dy in tl.static_range(0, POOL):
                    for dx in tl.static_range(0, POOL):
                        ih = base_oh + dy + kh
                        iw = base_ow + dx + kw
                        x_off = ((pid_n * IC + ic) * H + ih) * W + iw
                        x_m = p_mask & (ih < H) & (iw < W)
                        xv = tl.load(x_ptr + x_off, mask=x_m, other=0.0)
                        x_acc += xv

                pool_sum += wv[:, None] * x_acc[None, :]

    pooled = pool_sum * inv_pool2
    sig = tl.sigmoid(pooled)
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

        BLOCK_P = 128
        BLOCK_OC = 32
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
            num_warps=8,
            num_stages=2,
        )
        return out