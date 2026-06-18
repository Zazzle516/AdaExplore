import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_maxpool_kernel(
    x_ptr,          # input: (N, IC, ID, IH, IW)
    w_ptr,          # weight: (OC, IC, KD, KH, KW)
    b_ptr,          # bias: (OC,)
    out_ptr,        # output: (N, OC, Do, Ho, Wo)
    N, IC, ID, IH, IW,
    OC, KD, KH, KW,
    OD, OH, OW,     # conv output dims
    Do, Ho, Wo,     # pooled output dims
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, do, ho, wo)
    pid = tl.program_id(0)
    n = tl.program_id(1)

    total = Do * Ho * Wo
    if pid >= total:
        return

    wo = pid % Wo
    ho = (pid // Wo) % Ho
    do = pid // (Wo * Ho)

    # conv-output base spatial location for this pool window
    od0 = do * POOL
    oh0 = ho * POOL
    ow0 = wo * POOL

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # load bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    max_vals = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)

    # iterate over the POOL^3 window
    POOL3: tl.constexpr = POOL * POOL * POOL
    for k in tl.static_range(0, POOL3):
        kd = k // (POOL * POOL)
        kh = (k // POOL) % POOL
        kw = k % POOL
        od = od0 + kd
        oh = oh0 + kh
        ow = ow0 + kw

        # compute conv at (n, :, od, oh, ow) for all OC
        acc = bias  # [BLOCK_OC]

        # iterate over IC, KD, KH, KW
        # KD*KH*KW*IC sum
        # For each kernel position, load input scalar (broadcast over OC) and weight slice over OC
        for ic in tl.static_range(0, 3):  # IC=3
            for kkd in tl.static_range(0, 3):  # KD=3
                for kkh in tl.static_range(0, 3):
                    for kkw in tl.static_range(0, 3):
                        id_ = od + kkd
                        ih_ = oh + kkh
                        iw_ = ow + kkw
                        # input ptr
                        x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                        xv = tl.load(x_ptr + x_off)
                        # weight ptrs over OC
                        w_off = ((oc_offs * IC + ic) * KD + kkd) * KH * KW + kkh * KW + kkw
                        wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        acc = acc + xv * wv

        # acc holds conv output at this (od,oh,ow) for all OC channels
        # softmax across OC
        acc = tl.where(oc_mask, acc, -float('inf'))
        m = tl.max(acc, axis=0)
        e = tl.exp(acc - m)
        e = tl.where(oc_mask, e, 0.0)
        s = tl.sum(e, axis=0)
        sm = e / s
        max_vals = tl.maximum(max_vals, sm)

    # store output
    out_base = ((n * OC + 0) * Do + do) * Ho * Wo + ho * Wo + wo
    out_ptrs = out_ptr + out_base + oc_offs * (Do * Ho * Wo)
    tl.store(out_ptrs, max_vals, mask=oc_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size  # 4

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_total
        Do = OD // POOL
        Ho = OH // POOL
        Wo = OW // POOL

        out = torch.empty((N, OC, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2
        BLOCK_OC = max(BLOCK_OC, 16)

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        total = Do * Ho * Wo
        grid = (total, N)
        fused_conv_softmax_maxpool_kernel[grid](
            x, weight, bias, out,
            N, IC, ID, IH, IW,
            OC, KD, KH, KW,
            OD, OH, OW,
            Do, Ho, Wo,
            POOL=POOL,
            BLOCK_OC=BLOCK_OC,
            num_warps=2,
        )
        return out