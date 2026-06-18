import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + BN + AvgPool(4) (two stacked AvgPool2)
# pool(BN(convT(x) + conv_bias)) = scale * pool(convT(x)) + (scale*conv_bias + bn_shift)
#
# Strategy:
#  - One program computes BLOCK_OC output channels for one pooled voxel (n, pod, poh, pow)
#  - Pooled voxel covers 4x4x4 conv-output positions; for each, accumulate convT contribution
#    convT(x)[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
#    where id = (od + PD - kd) / SD if divisible & in range, etc.
#
# Optimization: with SD=SH=SW=2, KD=KH=KW=3, PD=PH=PW=1, for each (od mod 2),
# the valid kd values are { kd : (od+1-kd) even & in [0,3) & id in [0,ID) }.
# We just static-unroll and let the 'if valid' branches resolve at compile time
# via constexpr-friendly conditions where possible.
#
# Reorder loops: ic outer (load x once per (ic, id, ih, iw)), accumulate over OC via w broadcast.


@triton.jit
def _fused_kernel(
    x_ptr, w_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    POD, POH, POW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    KD_C: tl.constexpr, KH_C: tl.constexpr, KW_C: tl.constexpr,
    IC_C: tl.constexpr,
):
    pid = tl.program_id(0)
    pw = pid % POW
    t = pid // POW
    ph = t % POH
    t = t // POH
    pd = t % POD
    n = t // POD

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    od_base = pd * 4
    oh_base = ph * 4
    ow_base = pw * 4

    # Loop ic outer so we load weights once per (ic, kd, kh, kw) for all 64 positions
    for ic in tl.static_range(0, IC_C):
        for kd in tl.static_range(0, KD_C):
            for kh in tl.static_range(0, KH_C):
                for kw in tl.static_range(0, KW_C):
                    w_off = ((ic * OC + oc_offs) * KD_C + kd) * KH_C * KW_C + kh * KW_C + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                    x_sum = tl.zeros((1,), dtype=tl.float32)
                    # local scalar accumulator over the 4x4x4 valid positions for this (kd,kh,kw)
                    x_acc = 0.0
                    for dd in tl.static_range(0, 4):
                        od = od_base + dd
                        id_num = od + PD - kd
                        id_ = id_num // SD
                        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_ < ID) & (id_ >= 0)
                        for hh in tl.static_range(0, 4):
                            oh = oh_base + hh
                            ih_num = oh + PH - kh
                            ih = ih_num // SH
                            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih < IH) & (ih >= 0)
                            for ww in tl.static_range(0, 4):
                                ow = ow_base + ww
                                iw_num = ow + PW - kw
                                iw = iw_num // SW
                                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw < IW) & (iw >= 0)
                                valid = id_valid & ih_valid & iw_valid
                                x_off = ((n * IC_C + ic) * ID + id_) * IH * IW + ih * IW + iw
                                x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                x_acc += x_val

                    acc += x_acc * w_val

    acc = acc * (1.0 / 64.0)
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)
    out_val = acc * scale + shift

    out_off = ((n * OC + oc_offs) * POD + pd) * POH * POW + ph * POW + pw
    tl.store(out_ptr + out_off, out_val, mask=oc_mask)


def fused_convt_bn_pool(x, weight, scale, shift, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding
    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW
    POD = OD // 4
    POH = OH // 4
    POW = OW // 4

    out = torch.empty((N, OC, POD, POH, POW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2
    if BLOCK_OC < 16:
        BLOCK_OC = 16

    grid = (N * POD * POH * POW,)
    _fused_kernel[grid](
        x, weight, scale, shift, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        POD, POH, POW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        KD_C=KD, KH_C=KH, KW_C=KW,
        IC_C=IC,
        num_warps=1,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias

        if self.training:
            y = F.conv_transpose3d(x, weight, conv_bias, stride=self.stride, padding=self.padding)
            y = self.batch_norm(y)
            y = F.avg_pool3d(y, 2)
            y = F.avg_pool3d(y, 2)
            return y
        else:
            rm = self.batch_norm.running_mean
            rv = self.batch_norm.running_var
            eps = self.batch_norm.eps
            bw = self.batch_norm.weight
            bb = self.batch_norm.bias
            invstd = torch.rsqrt(rv + eps)
            scale = bw * invstd  # (OC,)
            if conv_bias is not None:
                shift = bb - rm * scale + scale * conv_bias
            else:
                shift = bb - rm * scale
            return fused_convt_bn_pool(x, weight, scale.contiguous(), shift.contiguous(),
                                       self.stride, self.padding)