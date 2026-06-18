import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,  # conv output dims
    PD, PH, PW,      # pooled output dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    WIN: tl.constexpr,   # pool window total = pool*pool
    IC_C: tl.constexpr,  # in_channels (constexpr)
    OC_C: tl.constexpr,  # out_channels (constexpr)
):
    # one program per (n, pd, ph, pw)
    pid = tl.program_id(0)
    pw = pid % PW
    t1 = pid // PW
    ph = t1 % PH
    t2 = t1 // PH
    pd = t2 % PD
    n = t2 // PD

    od_start = pd * WIN
    oh_start = ph * WIN
    ow_start = pw * WIN

    c_offs = tl.arange(0, OC_C)
    c_mask = c_offs < OC
    ic_offs = tl.arange(0, IC_C)
    ic_mask = ic_offs < IC

    # bias
    bias = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0)

    max_vals = tl.full((OC_C,), -float('inf'), dtype=tl.float32)

    # Iterate over WIN^3 conv output positions in the pool window
    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                od = od_start + dd
                oh = oh_start + hh
                ow = ow_start + ww

                # compute conv output for (n, :, od, oh, ow)
                acc = tl.zeros((OC_C,), dtype=tl.float32)

                # accumulate over kd, kh, kw, ic
                for kd in tl.static_range(0, KD):
                    id_ = od + kd
                    for kh in tl.static_range(0, KH):
                        ih = oh + kh
                        for kw in tl.static_range(0, KW):
                            iw = ow + kw
                            # x[n, ic, id_, ih, iw] for all ic
                            x_base = ((n * IC) * ID + id_) * IH * IW + ih * IW + iw
                            x_ptrs = x_ptr + x_base + ic_offs * (ID * IH * IW)
                            x_vals = tl.load(x_ptrs, mask=ic_mask, other=0.0)  # [IC_C]

                            # weight[oc, ic, kd, kh, kw] for all oc, all ic
                            # weight layout: (OC, IC, KD, KH, KW)
                            w_base = (kd * KH + kh) * KW + kw
                            # ptrs[oc, ic] = w_ptr + oc*(IC*KD*KH*KW) + ic*(KD*KH*KW) + w_base
                            w_ptrs = (w_ptr
                                      + c_offs[:, None] * (IC * KD * KH * KW)
                                      + ic_offs[None, :] * (KD * KH * KW)
                                      + w_base)
                            w_vals = tl.load(w_ptrs,
                                             mask=c_mask[:, None] & ic_mask[None, :],
                                             other=0.0)  # [OC_C, IC_C]

                            # acc[oc] += sum_ic w_vals[oc, ic] * x_vals[ic]
                            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

                acc = acc + bias
                # masked invalid OC to -inf for softmax max
                acc_for_sm = tl.where(c_mask, acc, -float('inf'))
                m = tl.max(acc_for_sm, axis=0)
                shifted = acc_for_sm - m
                e = tl.exp(shifted)
                e = tl.where(c_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                sm = tl.where(c_mask, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

    # Store output[n, c, pd, ph, pw]
    out_base = ((n * OC) * PD + pd) * PH * PW + ph * PW + pw
    out_ptrs = out_ptr + out_base + c_offs * (PD * PH * PW)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_conv_softmax_pool(x, weight, bias, pool_kernel_size):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    WIN = pool_kernel_size * pool_kernel_size
    PD = OD // WIN
    PH = OH // WIN
    PW = OW // WIN

    out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

    def npot(v):
        r = 1
        while r < v:
            r *= 2
        return r

    IC_C = npot(IC)
    OC_C = npot(OC)

    grid = (N * PD * PH * PW,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        PD, PH, PW,
        KD=KD, KH=KH, KW=KW,
        WIN=WIN,
        IC_C=IC_C,
        OC_C=OC_C,
        num_warps=4,
        num_stages=2,
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
                base = ((n * C + 0) * D + d) * H * W + h * W + w
                ptrs = x_ptr + base + c_offs * (D * H * W)
                vals = tl.load(ptrs, mask=c_mask, other=-float('inf'))
                m = tl.max(vals, axis=0)
                vals_shift = vals - m
                e = tl.exp(vals_shift)
                e = tl.where(c_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                sm = tl.where(c_mask, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

    out_base = ((n * C + 0) * OD + od) * OH * OW + oh * OW + ow
    out_ptrs = out_ptr + out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_softmax_pool(x, pool_kernel_size):
    N, C, D, H, W = x.shape
    WIN = pool_kernel_size * pool_kernel_size
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
        N, C, D, H, W, OD, OH, OW,
        WIN=WIN, BLOCK_C=BLOCK_C,
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
        WIN = self.pool_kernel_size * self.pool_kernel_size
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        if (OD % WIN == 0) and (OH % WIN == 0) and (OW % WIN == 0):
            try:
                return fused_conv_softmax_pool(
                    x, self.conv.weight, self.conv.bias, self.pool_kernel_size
                )
            except Exception:
                pass

        x = self.conv(x)
        N, C, D, H, W = x.shape
        if (D % WIN == 0) and (H % WIN == 0) and (W % WIN == 0):
            return fused_softmax_pool(x.contiguous(), self.pool_kernel_size)
        x = torch.softmax(x, dim=1)
        x = F.max_pool3d(x, self.pool_kernel_size)
        x = F.max_pool3d(x, self.pool_kernel_size)
        return x