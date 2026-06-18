import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _convt3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    PD_OD: tl.constexpr, PD_OH: tl.constexpr, PD_OW: tl.constexpr,
    MD: tl.constexpr, MH: tl.constexpr, MW: tl.constexpr,
    CLAMP_MIN: tl.constexpr, CLAMP_MAX: tl.constexpr,
    INV_S: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    # one program per (n, oc)
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    # Pooled output dims
    PD_ = PD_OD
    PH_ = PD_OH
    PW_ = PD_OW

    # We need to compute, for each pooled location (pd, ph, pw):
    #   max over (md, mh, mw) in MD x MH x MW of conv_out[n, oc, pd*MD+md, ph*MH+mh, pw*MW+mw]
    # then average over PD_*PH_*PW_ and clamp.
    #
    # conv_out[n, oc, od, oh, ow] = bias[oc] + sum_{ic, kd, kh, kw} x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
    # where id*SD + (KD-1-kd) - PD = od  -> kd = od + PD - id*SD ; valid if 0<=kd<KD
    # i.e. id*SD = od + PD - kd, so id = (od + PD - kd)/SD, integer.
    #
    # We'll loop over pooled locations one at a time. PD_*PH_*PW_ = 8*16*16 = 2048 per (n,oc).
    # Inside, loop over the MD*MH*MW = 8 maxpool window, and for each compute one conv output value.
    # For each conv output value, loop over (ic, kd, kh, kw) and accumulate.

    bias = tl.load(b_ptr + oc)

    sum_pool = 0.0

    # Precompute base pointers
    # weight layout: [IC, OC, KD, KH, KW]
    # input layout: [N, IC, ID, IH, IW]

    for pd in range(0, PD_):
        for ph in range(0, PH_):
            for pw in range(0, PW_):
                max_val = -float("inf")
                for md in range(0, MD):
                    od = pd * MD + md
                    for mh in range(0, MH):
                        oh = ph * MH + mh
                        for mw in range(0, MW):
                            ow = pw * MW + mw

                            acc = bias
                            # Loop over kd, kh, kw
                            for kd in range(0, KD):
                                id_num = od + PD - kd
                                id_ = id_num // SD
                                id_valid = ((id_num - id_ * SD) == 0) & (id_ >= 0) & (id_ < ID)
                                for kh in range(0, KH):
                                    ih_num = oh + PH - kh
                                    ih_ = ih_num // SH
                                    ih_valid = ((ih_num - ih_ * SH) == 0) & (ih_ >= 0) & (ih_ < IH)
                                    for kw in range(0, KW):
                                        iw_num = ow + PW - kw
                                        iw_ = iw_num // SW
                                        iw_valid = ((iw_num - iw_ * SW) == 0) & (iw_ >= 0) & (iw_ < IW)
                                        valid = id_valid & ih_valid & iw_valid

                                        # Vectorized load over IC tile
                                        ic_offs = tl.arange(0, BLOCK_IC)
                                        ic_mask = ic_offs < IC

                                        x_off = (n * IC * ID * IH * IW
                                                 + ic_offs * ID * IH * IW
                                                 + id_ * IH * IW
                                                 + ih_ * IW
                                                 + iw_)
                                        w_off = (ic_offs * OC * KD * KH * KW
                                                 + oc * KD * KH * KW
                                                 + kd * KH * KW
                                                 + kh * KW
                                                 + kw)

                                        m = ic_mask & valid
                                        xv = tl.load(x_ptr + x_off, mask=m, other=0.0)
                                        wv = tl.load(w_ptr + w_off, mask=m, other=0.0)
                                        acc += tl.sum(xv * wv, axis=0)

                            if acc > max_val:
                                max_val = acc

                sum_pool += max_val

    mean = sum_pool * INV_S
    mean = tl.minimum(tl.maximum(mean, CLAMP_MIN), CLAMP_MAX)
    tl.store(out_ptr + n * OC + oc, mean)


@triton.jit
def _final_reduce_kernel(
    in_ptr, out_ptr,
    N, C, S,
    scale,
    clamp_min, clamp_max,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = n * C * S + c * S

    acc = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        x = tl.load(in_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)

    mean = acc / S
    mean = mean * scale
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = float(scale)
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.maxpool_kernel_size = maxpool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        x = x.contiguous()
        N, C, D, H, W = x.shape
        S = D * H * W
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        grid = (N * C,)
        BLOCK = 1024
        _final_reduce_kernel[grid](
            x, out,
            N, C, S,
            float(self.scale),
            float(self.clamp_min), float(self.clamp_max),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out