import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, H, W,
    OC, OH, OW, PH, PW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    POOL: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, num_pool_tiles, num_oc_tiles)
    n = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_oc = tl.program_id(2)

    P_total = PH * PW
    p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P_total
    ph = p_offs // PW
    pw = p_offs % PW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Accumulator: [BLOCK_P, BLOCK_OC]
    acc = tl.zeros((BLOCK_P, BLOCK_OC), dtype=tl.float32)

    inv_pool2 = 1.0 / (POOL * POOL)

    # For each kernel position (kh, kw) and each pool offset (dy, dx),
    # accumulate x[n, ic, ph*POOL+dy+kh, pw*POOL+dx+kw] * w[oc, ic, kh, kw]
    # Loop over pool dy, dx, kh, kw: total POOL*POOL*KH*KW iterations.
    # For each, compute over all IC at once via dot.

    # We will iterate over (dy, dx, kh, kw) and for each load:
    #   x_block of shape [BLOCK_P, IC]: input at the position
    #   w_block of shape [IC, BLOCK_OC]: weight at (kh, kw, all ic)
    # then acc += dot(x_block, w_block)

    ic_range = tl.arange(0, IC)

    for dy in tl.static_range(POOL):
        for dx in tl.static_range(POOL):
            for kh in tl.static_range(KH):
                for kw in tl.static_range(KW):
                    in_h = ph * POOL + dy + kh  # [BLOCK_P]
                    in_w = pw * POOL + dx + kw  # [BLOCK_P]

                    # x indexing: [BLOCK_P, IC]
                    x_off = (n * stride_xn
                             + ic_range[None, :] * stride_xc
                             + in_h[:, None] * stride_xh
                             + in_w[:, None] * stride_xw)
                    x_block = tl.load(x_ptr + x_off,
                                      mask=p_mask[:, None],
                                      other=0.0)

                    # w indexing: weight shape is (OC, IC, KH, KW), contiguous
                    # w[oc_offs, :, kh, kw] -> [IC, BLOCK_OC]
                    w_off = (oc_offs[None, :] * (IC * KH * KW)
                             + ic_range[:, None] * (KH * KW)
                             + kh * KW + kw)
                    w_block = tl.load(w_ptr + w_off,
                                      mask=oc_mask[None, :],
                                      other=0.0)

                    acc += tl.dot(x_block, w_block)

    # Add bias scaled by POOL*POOL (since bias added per output, summed POOL*POOL times in pool window)
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bias[None, :] * (POOL * POOL)
    acc = acc * inv_pool2
    acc = tl.sigmoid(acc)

    # Mask invalid positions
    valid = p_mask[:, None] & oc_mask[None, :]
    acc = tl.where(valid, acc, 0.0)

    # Sum over pool tile (axis 0) -> [BLOCK_OC]
    partial = tl.sum(acc, axis=0)

    # Atomic add to out[n, oc_offs]
    out_ptrs = out_ptr + n * OC + oc_offs
    tl.atomic_add(out_ptrs, partial, mask=oc_mask)


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
        OH = H - KH + 1
        OW = W - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        buf = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_P = 64
        BLOCK_OC = 64

        num_p_tiles = (PH * PW + BLOCK_P - 1) // BLOCK_P
        num_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC

        grid = (N, num_p_tiles, num_oc_tiles)

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, buf,
            N, H, W,
            OC, OH, OW, PH, PW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            POOL=POOL,
            KH=KH, KW=KW, IC=IC,
            BLOCK_P=BLOCK_P,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        return buf.sum(dim=1)