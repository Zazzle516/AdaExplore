import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + BN + AvgPool(2) + AvgPool(2) = AvgPool(4)
# One program computes one pooled output voxel: (n, oc, pod, poh, pow_tile)
# For each pooled voxel we sum 4x4x4=64 conv-transpose outputs, then /64, then *scale+shift_post
# where scale = bn_weight * invstd, shift_post = (bn_bias - rm*scale) (constant added per voxel * 1 = shift)
# Actually after pooling avg, BN affine applied AFTER conv but BEFORE pool. Since BN affine is linear:
# pool(BN(y)) = pool(scale*y + shift) = scale * pool(y) + shift
# So we can compute pool(y) (where y is conv-transpose output, no bias yet),
# then apply final affine. Also conv bias: y_with_bias = y + bias_conv, pool gives pool(y) + bias_conv.
# So final = scale * (pool(y) + bias_conv) + shift = scale * pool(y) + (scale*bias_conv + shift)

# Implementation: gather-style. For each pooled output voxel:
#   accumulate over 4x4x4 conv-output positions (od, oh, ow)
#   For each (od, oh, ow), the conv-transpose output is:
#     sum over (ic, kd, kh, kw) of x[n, ic, (od+PD-kd)/SD, ...] * w[ic, oc, kd, kh, kw]
#     where the indices must be divisible by stride and within [0, ID)
#
# Reorganize: pre-sum over the 64 pool positions: out[n,oc,pod,poh,pow] = (1/64) *
#   sum_{dd,hh,ww in 0..3} sum_{ic,kd,kh,kw} x[n,ic, id, ih, iw] * w[ic,oc,kd,kh,kw]
# where id = (pod*4 + dd + PD - kd) / SD if divisible

# Better: for each (ic, kd, kh, kw), and for each (dd, hh, ww), determine input index and accumulate.
# Sum over OC tile in registers.


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
    # pid -> (n, pod, poh, pow)
    pw = pid % POW
    t = pid // POW
    ph = t % POH
    t = t // POH
    pd = t % POD
    n = t // POD

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # base output coordinates (conv output) for this pooled voxel
    od_base = pd * 4
    oh_base = ph * 4
    ow_base = pw * 4

    # Iterate over the 64 conv-output positions in this pooled voxel
    for dd in tl.static_range(0, 4):
        od = od_base + dd
        for hh in tl.static_range(0, 4):
            oh = oh_base + hh
            for ww in tl.static_range(0, 4):
                ow = ow_base + ww
                # For each conv-output (od, oh, ow), accumulate conv-transpose
                # over (ic, kd, kh, kw)
                for kd in tl.static_range(0, KD_C):
                    id_num = od + PD - kd
                    id_ = id_num // SD
                    id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_ < ID)
                    for kh in tl.static_range(0, KH_C):
                        ih_num = oh + PH - kh
                        ih = ih_num // SH
                        ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih < IH)
                        for kw in tl.static_range(0, KW_C):
                            iw_num = ow + PW - kw
                            iw = iw_num // SW
                            iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw < IW)
                            valid = id_valid & ih_valid & iw_valid
                            if valid:
                                for ic in tl.static_range(0, IC_C):
                                    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                                    x_val = tl.load(x_ptr + x_off)
                                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                    acc += x_val * w_val

    acc = acc * (1.0 / 64.0)
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)
    out_val = acc * scale + shift

    out_off = ((n * OC + oc_offs) * POD + pd) * POH * POW + ph * POW + pw
    tl.store(out_ptr + out_off, out_val, mask=oc_mask)


def fused_convt_bn_pool(x, weight, conv_bias, bn_scale, bn_shift_pre, stride, padding):
    # bn_scale = bn_w * invstd
    # bn_shift = bn_b - rm * bn_scale
    # final shift_post = bn_scale * conv_bias + bn_shift  (since conv bias is added BEFORE BN)
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

    BLOCK_OC = 16
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = (N * POD * POH * POW,)
    _fused_kernel[grid](
        x, weight, bn_scale, bn_shift_pre, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        POD, POH, POW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        KD_C=KD, KH_C=KH, KW_C=KW,
        IC_C=IC,
        num_warps=2,
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
            return fused_convt_bn_pool(x, weight, conv_bias, scale.contiguous(), shift.contiguous(),
                                        self.stride, self.padding)