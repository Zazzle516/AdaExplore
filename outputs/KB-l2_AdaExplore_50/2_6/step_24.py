import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_maxpool_kernel(
    x_ptr,       # input: (N, IC, ID, IH, IW)
    w_ptr,       # weight: (OC, IC, KD, KH, KW)
    b_ptr,       # bias: (OC,)
    out_ptr,     # output: (N, OC, Do, Ho, Wo)
    N, IC, ID, IH, IW,
    OC,
    OD, OH, OW,  # conv output spatial sizes
    Do, Ho, Wo,  # pooled output spatial sizes
    BLOCK_OC: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    IC_C: tl.constexpr,
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

    d0 = do * 4
    h0 = ho * 4
    w0 = wo * 4

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Load bias once
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # max over the 4x4x4 window of softmax-normalized conv outputs
    max_vals = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)

    # iterate over 4x4x4 = 64 spatial positions
    for k in tl.static_range(0, 64):
        dk = k // 16
        hk = (k // 4) % 4
        wk = k % 4
        od = d0 + dk
        oh = h0 + hk
        ow = w0 + wk

        in_bounds = (od < OD) & (oh < OH) & (ow < OW)

        # compute conv output at (n, :, od, oh, ow) for all OC
        acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

        # Loop over IC * KD * KH * KW = 3*3*3*3 = 81
        for kk in tl.static_range(0, IC_C * KD * KH * KW):
            ic = kk // (KD * KH * KW)
            rem = kk % (KD * KH * KW)
            kd = rem // (KH * KW)
            rem2 = rem % (KH * KW)
            kh = rem2 // KW
            kw = rem2 % KW

            id_ = od + kd
            ih = oh + kh
            iw = ow + kw

            # load input scalar
            x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
            x_val = tl.load(x_ptr + x_off, mask=in_bounds, other=0.0)

            # load weight column for this (ic, kd, kh, kw) across all OC
            w_off = oc_offs * (IC_C * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
            w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

            acc += x_val * w_vals

        acc = acc + bias

        # softmax across OC at this spatial position
        # mask invalid OC to -inf
        acc_masked = tl.where(oc_mask, acc, -float('inf'))
        m = tl.max(acc_masked, axis=0)
        e = tl.exp(acc_masked - m)
        e = tl.where(oc_mask, e, 0.0)
        s = tl.sum(e, axis=0)
        sm = e / s

        # for out-of-bounds spatial positions, set sm to -inf
        sm = tl.where(in_bounds, sm, -float('inf'))
        max_vals = tl.maximum(max_vals, sm)

    # store: out[n, :, do, ho, wo]
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
        self.pool_total = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        pt = self.pool_total
        Do = OD // pt
        Ho = OH // pt
        Wo = OW // pt

        out = torch.empty((N, OC, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        BLOCK_OC = 16
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        total = Do * Ho * Wo
        grid = (total, N)
        fused_conv_softmax_maxpool_kernel[grid](
            x, weight, bias, out,
            N, IC, ID, IH, IW,
            OC,
            OD, OH, OW,
            Do, Ho, Wo,
            BLOCK_OC=BLOCK_OC,
            KD=KD, KH=KH, KW=KW,
            IC_C=IC,
            num_warps=2,
        )
        return out