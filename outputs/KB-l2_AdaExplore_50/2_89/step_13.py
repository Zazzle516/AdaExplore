import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,  # pooled output dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N * PD * PH * PW, )
    pid = tl.program_id(0)
    pw = pid % PW
    tmp = pid // PW
    ph = tmp % PH
    tmp = tmp // PH
    pd = tmp % PD
    n = tmp // PD

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # pooled output covers output indices [pd*2..pd*2+1] x [ph*2..ph*2+1] x [pw*2..pw*2+1]
    # for each, compute conv transpose output, then take max over pool window
    acc_max = tl.full([BLOCK_OC], -float('inf'), dtype=tl.float32)

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)

    for ld in tl.static_range(POOL_K):
        for lh in tl.static_range(POOL_K):
            for lw in tl.static_range(POOL_K):
                od = pd * POOL_K + ld
                oh = ph * POOL_K + lh
                ow = pw * POOL_K + lw

                # Compute conv transpose value at (n, :, od, oh, ow)
                # out[n, oc, od, oh, ow] = sum over ic, kd, kh, kw:
                #   x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
                # where id*SD - PAD_D + kd = od, etc.
                # So kd = od + PAD_D - id*SD, must be in [0, KD)
                # i.e. id = (od + PAD_D - kd) / SD, integer and in [0, ID)

                acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

                for kd in tl.static_range(KD):
                    id_num = od + PAD_D - kd
                    id_ = id_num // SD
                    id_valid = (id_num % SD == 0) & (id_ >= 0) & (id_ < ID)
                    for kh in tl.static_range(KH):
                        ih_num = oh + PAD_H - kh
                        ih_ = ih_num // SH
                        ih_valid = (ih_num % SH == 0) & (ih_ >= 0) & (ih_ < IH)
                        for kw in tl.static_range(KW):
                            iw_num = ow + PAD_W - kw
                            iw_ = iw_num // SW
                            iw_valid = (iw_num % SW == 0) & (iw_ >= 0) & (iw_ < IW)
                            valid = id_valid & ih_valid & iw_valid

                            for ic in tl.static_range(0, 3):
                                # load x[n, ic, id_, ih_, iw_]
                                x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                # load w[ic, :, kd, kh, kw] : shape [OC]
                                w_off = ((ic * OC + offs_oc) * KD + kd) * KH * KW + kh * KW + kw
                                wv = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                                acc += xv * wv

                acc = acc + bias
                acc_max = tl.maximum(acc_max, acc)

    # store pooled result: shape (N, OC, PD, PH, PW)
    out_off = ((n * OC + offs_oc) * PD + pd) * PH * PW + ph * PW + pw
    tl.store(out_ptr + out_off, acc_max, mask=mask_oc)


@triton.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * S + s
    x_ptrs = x_ptr + base + offs_c * S

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * S + s, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PAD_D = PAD_H = PAD_W = self.padding
        OPD = OPH = OPW = self.output_padding

        # output spatial dims for conv transpose
        OD = (ID - 1) * SD - 2 * PAD_D + KD + OPD
        OH = (IH - 1) * SH - 2 * PAD_H + KH + OPH
        OW = (IW - 1) * SW - 2 * PAD_W + KW + OPW

        # pooled dims
        POOL_K = self.pool_kernel_size
        PD = OD // POOL_K
        PH = OH // POOL_K
        PW = OW // POOL_K

        # Fused conv_transpose + maxpool: produce (N, OC, PD, PH, PW)
        pooled = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=x.dtype)

        w = self.conv_transpose.weight.contiguous()  # (IC, OC, KD, KH, KW)
        b = self.conv_transpose.bias.contiguous()    # (OC,)

        BLOCK_OC = triton.next_power_of_2(OC)
        grid = (N * PD * PH * PW,)
        conv_transpose_pool_kernel[grid](
            x, w, b, pooled,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            PD, PH, PW,
            KD, KH, KW,
            SD, SH, SW,
            PAD_D, PAD_H, PAD_W,
            POOL_K,
            BLOCK_OC=BLOCK_OC,
            num_warps=2,
        )

        # Softmax + sub + swish + channel-max
        S = PD * PH * PW
        out = torch.empty((N, PD, PH, PW), device=x.device, dtype=x.dtype)
        BLOCK_C = triton.next_power_of_2(OC)
        grid2 = (N * S,)
        fused_softmax_sub_swish_max_kernel[grid2](
            pooled, self.subtract.contiguous(), out,
            N, OC, S,
            BLOCK_C=BLOCK_C,
            num_warps=1,
        )
        return out