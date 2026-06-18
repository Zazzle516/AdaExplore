import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr,        # input: (N, IC, ID, IH, IW)
    w_ptr,        # weight: (OC, IC, KD, KH, KW)
    b_ptr,        # bias: (OC,)
    out_ptr,      # output: (N, OC, OD, OH, OW)
    N, IC, ID, IH, IW,
    OC,
    CD, CH, CW,   # conv output spatial dims
    OD, OH, OW,   # final output spatial dims (after pool)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    WIN: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    pid1 = pid // OW
    oh = pid1 % OH
    pid2 = pid1 // OH
    od = pid2 % OD
    n = pid2 // OD

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Pool window in conv-output space
    cd_start = od * WIN
    ch_start = oh * WIN
    cw_start = ow * WIN

    # Load bias once
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Accumulator: per-channel max of softmax values
    max_vals = tl.full((BLOCK_OC,), -float('inf'), dtype=tl.float32)

    # Iterate over the WIN^3 positions in conv-output space
    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                cd = cd_start + dd
                ch = ch_start + hh
                cw = cw_start + ww

                # Compute conv output for all OC channels at (n, :, cd, ch, cw)
                acc = bias  # start from bias

                # Sum over IC, KD, KH, KW
                for ic in tl.static_range(0, 3):  # IC=3
                    for kd in tl.static_range(0, KD):
                        for kh in tl.static_range(0, KH):
                            for kw in tl.static_range(0, KW):
                                id_ = cd + kd
                                ih_ = ch + kh
                                iw_ = cw + kw
                                # Load input scalar
                                in_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                                x_val = tl.load(x_ptr + in_off)
                                # Load weight vector over OC
                                # weight shape (OC, IC, KD, KH, KW), stride OC=IC*KD*KH*KW
                                w_off = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                                w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc = acc + x_val * w_vals

                # Softmax over OC
                acc = tl.where(oc_mask, acc, -float('inf'))
                m = tl.max(acc, axis=0)
                shifted = acc - m
                e = tl.exp(shifted)
                e = tl.where(oc_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                sm = tl.where(oc_mask, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

    # Store output: out[n, oc, od, oh, ow]
    out_base = ((n * OC + 0) * OD + od) * OH * OW + oh * OW + ow
    out_ptrs = out_ptr + out_base + oc_offs * (OD * OH * OW)
    tl.store(out_ptrs, max_vals, mask=oc_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_kernel_size):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    CD = ID - KD + 1
    CH = IH - KH + 1
    CW = IW - KW + 1
    POOL = pool_kernel_size
    WIN = POOL * POOL
    OD = CD // WIN
    OH = CH // WIN
    OW = CW // WIN

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = (N * OD * OH * OW,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC,
        CD, CH, CW,
        OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        WIN=WIN,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
    )
    return out


@triton.jit
def fused_softmax_pool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    WIN: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    pid1 = pid // OW
    oh = pid1 % OH
    pid2 = pid1 // OH
    od = pid2 % OD
    n = pid2 // OD

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    d_start = od * WIN
    h_start = oh * WIN
    w_start = ow * WIN

    max_vals = tl.full((BLOCK_C,), -float('inf'), dtype=tl.float32)

    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                d = d_start + dd
                h = h_start + hh
                w = w_start + ww
                in_bounds = (d < D) & (h < H) & (w < W)
                base = ((n * C + 0) * D + d) * H * W + h * W + w
                ptrs = x_ptr + base + c_offs * (D * H * W)
                vals = tl.load(ptrs, mask=c_mask & in_bounds, other=-float('inf'))
                m = tl.max(vals, axis=0)
                vals_shift = vals - m
                e = tl.exp(vals_shift)
                e = tl.where(c_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                sm = tl.where(c_mask & in_bounds, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

    out_base = ((n * C + 0) * OD + od) * OH * OW + oh * OW + ow
    out_ptrs = out_ptr + out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_softmax_pool(x, pool_kernel_size):
    N, C, D, H, W = x.shape
    POOL = pool_kernel_size
    WIN = POOL * POOL
    OD = D // WIN
    OH = H // WIN
    OW = W // WIN

    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * OD * OH * OW,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        WIN=WIN,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        CD = ID - KD + 1
        CH = IH - KH + 1
        CW = IW - KW + 1
        WIN = self.pool_kernel_size * self.pool_kernel_size

        if (IC == 3 and self.out_channels == 16 and KD == 3 and KH == 3 and KW == 3
                and CD % WIN == 0 and CH % WIN == 0 and CW % WIN == 0):
            return fused_conv_softmax_pool(x, weight, bias, self.pool_kernel_size)
        else:
            x = self.conv(x)
            N, C, D, H, W = x.shape
            if D % WIN == 0 and H % WIN == 0 and W % WIN == 0:
                return fused_softmax_pool(x.contiguous(), self.pool_kernel_size)
            else:
                x = torch.softmax(x, dim=1)
                x = F.max_pool3d(x, self.pool_kernel_size)
                x = F.max_pool3d(x, self.pool_kernel_size)
                return x