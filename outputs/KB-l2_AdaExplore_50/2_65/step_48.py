import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    PH, PW,
    POOL: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC_C: tl.constexpr,
    OC_BLOCK: tl.constexpr,
    P_BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    oc_offs = pid_oc * OC_BLOCK + tl.arange(0, OC_BLOCK)
    oc_mask = oc_offs < OC

    p_offs = pid_p * P_BLOCK + tl.arange(0, P_BLOCK)
    p_mask = p_offs < (PH * PW)
    ph = p_offs // PW
    pw = p_offs % PW

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    inv_pool2 = 1.0 / (POOL * POOL)

    # accumulator over pooled cells: [OC_BLOCK, P_BLOCK]
    # We compute conv outputs for all POOL*POOL positions per pool cell,
    # accumulating the sum (then multiply by inv_pool2 at the end).
    pooled = tl.zeros([OC_BLOCK, P_BLOCK], dtype=tl.float32)

    # Total kernel positions per pool cell: POOL*POOL conv positions,
    # each needing IC*KH*KW MACs. We restructure as:
    # for each (ic, kh, kw): for each (dy, dx) in pool: load x, accumulate.
    # Weight is loaded once per (ic, kh, kw) for OC_BLOCK.
    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                w_off = ((oc_offs * IC) + ic) * KH * KW + kh * KW + kw
                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [OC_BLOCK]

                # Sum x over pool window for fixed (kh, kw)
                x_sum = tl.zeros([P_BLOCK], dtype=tl.float32)
                for dy in tl.static_range(0, POOL):
                    for dx in tl.static_range(0, POOL):
                        oh = ph * POOL + dy
                        ow = pw * POOL + dx
                        ih = oh + kh
                        iw = ow + kw
                        x_off = ((pid_n * IC + ic) * H + ih) * W + iw
                        xv = tl.load(x_ptr + x_off, mask=p_mask, other=0.0)
                        x_sum += xv

                pooled += wv[:, None] * x_sum[None, :]

    # Add bias contribution: bias appears POOL*POOL times per pool cell
    pooled += bias[:, None] * (POOL * POOL)
    pooled = pooled * inv_pool2

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

        OC_BLOCK = 64
        P_BLOCK = 256

        n_oc_tiles = (OC + OC_BLOCK - 1) // OC_BLOCK
        n_p_tiles = (PH * PW + P_BLOCK - 1) // P_BLOCK

        grid = (N, n_oc_tiles, n_p_tiles)
        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, OH, OW, PH, PW,
            POOL=POOL,
            KH=KH, KW=KW,
            IC_C=IC,
            OC_BLOCK=OC_BLOCK,
            P_BLOCK=P_BLOCK,
            num_warps=8,
            num_stages=3,
        )
        return out