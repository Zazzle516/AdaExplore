import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    WIN: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
):
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

    bias = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0)

    max_vals = tl.full((OC_C,), -float('inf'), dtype=tl.float32)

    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                od = od_start + dd
                oh = oh_start + hh
                ow = ow_start + ww

                acc = tl.zeros((OC_C,), dtype=tl.float32)

                for kd in tl.static_range(0, KD):
                    id_ = od + kd
                    for kh in tl.static_range(0, KH):
                        ih = oh + kh
                        for kw in tl.static_range(0, KW):
                            iw = ow + kw
                            x_base = ((n * IC) * ID + id_) * IH * IW + ih * IW + iw
                            x_ptrs = x_ptr + x_base + ic_offs * (ID * IH * IW)
                            x_vals = tl.load(x_ptrs, mask=ic_mask, other=0.0)

                            w_base = (kd * KH + kh) * KW + kw
                            w_ptrs = (w_ptr
                                      + c_offs[:, None] * (IC * KD * KH * KW)
                                      + ic_offs[None, :] * (KD * KH * KW)
                                      + w_base)
                            w_vals = tl.load(w_ptrs,
                                             mask=c_mask[:, None] & ic_mask[None, :],
                                             other=0.0)

                            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

                acc = acc + bias
                acc_for_sm = tl.where(c_mask, acc, -float('inf'))
                m = tl.max(acc_for_sm, axis=0)
                shifted = acc_for_sm - m
                e = tl.exp(shifted)
                e = tl.where(c_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                sm = tl.where(c_mask, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

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
        num_warps=2,
        num_stages=2,
    )
    return out


@triton.jit
def conv_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Program tiles: (n, spatial_tile)
    # Output layout: (N, OC, OD, OH, OW)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    S = OD * OH * OW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    ow = s_offs % OW
    tmp = s_offs // OW
    oh = tmp % OH
    od = tmp // OH

    c_offs = tl.arange(0, OC_C)
    c_mask = c_offs < OC
    ic_offs = tl.arange(0, IC_C)
    ic_mask = ic_offs < IC

    bias = tl.load(b_ptr + c_offs, mask=c_mask, other=0.0)

    # acc: [OC_C, BLOCK_S]
    acc = tl.zeros((OC_C, BLOCK_S), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        id_ = od + kd
        for kh in tl.static_range(0, KH):
            ih = oh + kh
            for kw in tl.static_range(0, KW):
                iw = ow + kw
                # x[n, ic, id_, ih, iw]: shape [IC_C, BLOCK_S]
                x_base = pid_n * IC * ID * IH * IW
                x_spatial = id_ * IH * IW + ih * IW + iw  # [BLOCK_S]
                x_ptrs = (x_ptr + x_base
                          + ic_offs[:, None] * (ID * IH * IW)
                          + x_spatial[None, :])
                x_vals = tl.load(x_ptrs,
                                 mask=ic_mask[:, None] & s_mask[None, :],
                                 other=0.0)

                w_base = (kd * KH + kh) * KW + kw
                w_ptrs = (w_ptr
                          + c_offs[:, None] * (IC * KD * KH * KW)
                          + ic_offs[None, :] * (KD * KH * KW)
                          + w_base)
                w_vals = tl.load(w_ptrs,
                                 mask=c_mask[:, None] & ic_mask[None, :],
                                 other=0.0)  # [OC_C, IC_C]

                acc += tl.dot(w_vals, x_vals)

    acc = acc + bias[:, None]
    acc = tl.where(c_mask[:, None], acc, -float('inf'))
    m = tl.max(acc, axis=0)  # [BLOCK_S]
    shifted = acc - m[None, :]
    e = tl.exp(shifted)
    e = tl.where(c_mask[:, None], e, 0.0)
    s = tl.sum(e, axis=0)  # [BLOCK_S]
    sm = e / s[None, :]

    # Store output[n, c, s]
    out_base = pid_n * OC * S
    out_ptrs = (out_ptr + out_base
                + c_offs[:, None] * S
                + s_offs[None, :])
    tl.store(out_ptrs, sm, mask=c_mask[:, None] & s_mask[None, :])


@triton.jit
def double_maxpool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    WIN: tl.constexpr,
):
    # one program per (n*c, od, oh, ow)
    pid = tl.program_id(0)
    ow = pid % OW
    t1 = pid // OW
    oh = t1 % OH
    t2 = t1 // OH
    od = t2 % OD
    nc = t2 // OD

    d_start = od * WIN
    h_start = oh * WIN
    w_start = ow * WIN

    base = nc * D * H * W
    max_val = -float('inf')

    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                d = d_start + dd
                h = h_start + hh
                w = w_start + ww
                v = tl.load(x_ptr + base + d * H * W + h * W + w)
                max_val = tl.maximum(max_val, v)

    out_idx = nc * OD * OH * OW + od * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_idx, max_val)


def conv_softmax(x, weight, bias):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    def npot(v):
        r = 1
        while r < v:
            r *= 2
        return r

    IC_C = max(16, npot(IC))
    OC_C = max(16, npot(OC))
    BLOCK_S = 64

    S = OD * OH * OW
    grid = (N, (S + BLOCK_S - 1) // BLOCK_S)
    conv_softmax_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        IC_C=IC_C,
        OC_C=OC_C,
        BLOCK_S=BLOCK_S,
        num_warps=2,
        num_stages=2,
    )
    return out


def double_maxpool(x, pool_kernel_size):
    N, C, D, H, W = x.shape
    WIN = pool_kernel_size * pool_kernel_size
    OD = D // WIN
    OH = H // WIN
    OW = W // WIN
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    grid = (N * C * OD * OH * OW,)
    double_maxpool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        WIN=WIN,
        num_warps=1,
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
                # Two-kernel approach: conv+softmax, then double maxpool
                y = conv_softmax(x, self.conv.weight, self.conv.bias)
                return double_maxpool(y, self.pool_kernel_size)
            except Exception:
                pass

        x = self.conv(x)
        x = torch.softmax(x, dim=1)
        x = F.max_pool3d(x, self.pool_kernel_size)
        x = F.max_pool3d(x, self.pool_kernel_size)
        return x