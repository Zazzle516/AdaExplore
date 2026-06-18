import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-add based ConvTranspose3d.
# Input:  x (N, IC, ID, IH, IW)
# Weight: w (IC, OC, KD, KH, KW)
# Bias:   b (OC,)
# Output: y (N, OC, OD, OH, OW), OD = (ID-1)*S - 2P + KD
#
# For stride=2, padding=1, K=3: each input voxel (id,ih,iw) contributes to
# output positions (od, oh, ow) where od = id*2 + kd - 1, etc. and 0<=od<OD.
#
# Strategy: one program per (n, ic_tile, spatial_block of input). For each
# (kd,kh,kw) and each oc, scatter-add input*weight into the output.
#
# To avoid races between programs writing to the same output element, we make
# each program own a unique set of output elements by partitioning along the
# input spatial dim per-(kd,kh,kw). With stride=2, two different input voxels
# never map to the same output (od,oh,ow) for the same (kd,kh,kw), but they
# DO map to the same output element for different (kd,kh,kw) choices. So one
# program must handle all (kd,kh,kw) for the input voxels it owns; and two
# programs must not own input voxels whose 3x3x3 footprints overlap on the
# same output.
#
# Simpler safe approach: one program per (n, oc) accumulates the output by
# gathering — that's the previous gather kernel. Instead we do a fused
# pipeline: compute the conv_transpose output on the fly per *final* output
# element (after avg_pool 4x4x4 + BN), without sharing across pool neighbors.
#
# Per safety contract, each of the 64 underlying convT outputs in a pool
# window must execute its full convT multiply-add count. We arrange the
# loops so each (od,oh,ow) inside the 4x4x4 window does its own IC*KD*KH*KW
# accumulation.
#
# But: we can save the BN intermediate. Output: (N, OC, OD/4, OH/4, OW/4)
# = (64, 16, 16, 16, 16)  (since OD=OH=OW=64 -> 64/4=16... wait 64/4=16)
# Actually OD = (32-1)*2 - 2 + 3 = 63. Hmm not a multiple of 4.
# Let me recompute: (ID-1)*S - 2P + K = 31*2 - 2 + 3 = 63. So OD=63.
# After avg_pool(2)->31 (floor), then avg_pool(2)->15. So final is 15.
# Avg pool with kernel=2 default stride=2, floor((63)/2)=31, floor(31/2)=15.
# So effective is pool with kernel=2,stride=2 twice => taking 4 consecutive
# elements starting at 0,2,4,...,28, but pool2 then pool2 means index map:
#   first pool: out1[i] = (in[2i]+in[2i+1])/2, i in [0,31)
#   second pool: out2[j] = (out1[2j]+out1[2j+1])/2 = (in[4j]+in[4j+1]+in[4j+2]+in[4j+3])/4
# So final size = 15, sampling starting from in[0..3], in[4..7], ..., in[56..59].
# in[60..63] (4 elements) are unused since 15*4=60.
#
# So OD_final = OH_final = OW_final = 15, taking a 4x4x4 window aligned at
# (4*od_f, 4*oh_f, 4*ow_f).
#
# Total output elements: 64*16*15*15*15 = 3,456,000. That's small.
# Total ops per output element: 64 (pool) * 3 (IC) * 27 (KD*KH*KW) = 5184
# multiply-adds. Total work: ~1.8e10 ops. At RTX 4090 ~50 TFLOPS = ~0.36ms ideal.
#
# However the gather is masked (only ~1/8 valid for stride=2). Still 5184/8 = 648
# effective MACs per output, *3.46M outputs = 2.24e9 MACs. Very small. We'll
# fuse everything.

@triton.jit
def fused_convT_bn_pool_kernel(
    x_ptr, w_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    ODF, OHF, OWF,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    OC_C: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_S: tl.constexpr,   # final spatial tile
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    spatial_f = ODF * OHF * OWF
    offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs < spatial_f

    odf = offs // (OHF * OWF)
    rem = offs % (OHF * OWF)
    ohf = rem // OWF
    owf = rem % OWF

    # base coords in pre-pool grid
    od0 = odf * 4
    oh0 = ohf * 4
    ow0 = owf * 4

    oc_range = tl.arange(0, OC_C)
    ic_range = tl.arange(0, IC_C)
    mask_oc = oc_range < OC
    mask_ic = ic_range < IC

    # accumulator [OC_C, BLOCK_S]
    acc = tl.zeros((OC_C, BLOCK_S), dtype=tl.float32)

    # iterate pool window
    for dd in tl.static_range(0, 4):
        for hh in tl.static_range(0, 4):
            for ww in tl.static_range(0, 4):
                od = od0 + dd
                oh = oh0 + hh
                ow = ow0 + ww

                # convT accumulation over (ic, kd, kh, kw)
                for kd in tl.static_range(0, KD):
                    id_num = od + PD - kd
                    id_q = id_num // SD
                    valid_d = (id_num >= 0) & ((id_num - id_q * SD) == 0) & (id_q >= 0) & (id_q < ID)

                    for kh in tl.static_range(0, KH):
                        ih_num = oh + PH - kh
                        ih_q = ih_num // SH
                        valid_h = (ih_num >= 0) & ((ih_num - ih_q * SH) == 0) & (ih_q >= 0) & (ih_q < IH)

                        for kw in tl.static_range(0, KW):
                            iw_num = ow + PW - kw
                            iw_q = iw_num // SW
                            valid_w = (iw_num >= 0) & ((iw_num - iw_q * SW) == 0) & (iw_q >= 0) & (iw_q < IW)

                            valid = valid_d & valid_h & valid_w & mask_s  # [BLOCK_S]

                            # x[n, ic, id_q, ih_q, iw_q] for all ic -> [IC_C, BLOCK_S]
                            x_off = ((pid_n * IC + ic_range[:, None]) * ID + id_q[None, :]) * IH * IW + ih_q[None, :] * IW + iw_q[None, :]
                            x_mask = valid[None, :] & mask_ic[:, None]
                            x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [IC_C, BLOCK_S]

                            # weight[ic, oc, kd, kh, kw] -> [IC_C, OC_C]
                            w_off = ((ic_range[:, None] * OC + oc_range[None, :]) * KD + kd) * KH * KW + kh * KW + kw
                            w_mask = mask_ic[:, None] & mask_oc[None, :]
                            w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [IC_C, OC_C]

                            # acc[OC,BLOCK_S] += w_val^T @ x_val
                            acc += tl.dot(tl.trans(w_val), x_val)

    acc = acc * (1.0 / 64.0)

    # apply scale/shift (BN folded, includes conv bias absorbed)
    s = tl.load(scale_ptr + oc_range, mask=mask_oc, other=0.0)
    sh = tl.load(shift_ptr + oc_range, mask=mask_oc, other=0.0)
    acc = acc * s[:, None] + sh[:, None]

    # store: out[n, oc, odf, ohf, owf]
    out_base = pid_n * OC * spatial_f
    out_offs = out_base + oc_range[:, None] * spatial_f + offs[None, :]
    store_mask = mask_s[None, :] & mask_oc[:, None]
    tl.store(out_ptr + out_offs, acc, mask=store_mask)


def _next_pow2(x):
    r = 1
    while r < x:
        r *= 2
    return r


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

    def _get_fused_scale_shift(self, y_unused_dtype, device, dtype):
        bn = self.batch_norm
        eps = bn.eps
        var = bn.running_var
        mean = bn.running_mean
        gamma = bn.weight
        beta = bn.bias
        # y = gamma*(x-mean)/sqrt(var+eps) + beta
        # so scale = gamma / sqrt(var+eps), shift = beta - mean*scale
        # then we add conv bias into shift: x_pre_bn = convT_out + conv_bias
        # so after BN: scale*(convT_out + conv_bias) + shift
        # we can fold conv_bias into shift too: shift' = shift + scale*conv_bias
        scale = (gamma / torch.sqrt(var + eps)).to(dtype)
        conv_bias = self.conv_transpose.bias.to(dtype)
        shift = (beta - mean * (gamma / torch.sqrt(var + eps))).to(dtype) + scale * conv_bias
        return scale.contiguous(), shift.contiguous()

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        if self.training:
            # fall back: compute full pipeline with cuDNN + torch ops
            y = F.conv_transpose3d(x, weight, bias, stride=self.stride, padding=self.padding)
            y = self.batch_norm(y)
            y = F.avg_pool3d(y, 2)
            y = F.avg_pool3d(y, 2)
            return y

        # Final pooled sizes (two pool2 = effective pool4 sampling first 4 of every 4)
        ODF = (OD // 2) // 2
        OHF = (OH // 2) // 2
        OWF = (OW // 2) // 2

        scale, shift = self._get_fused_scale_shift(x.dtype, x.device, x.dtype)

        out = torch.empty((N, OC, ODF, OHF, OWF), device=x.device, dtype=x.dtype)

        spatial_f = ODF * OHF * OWF
        OC_C = _next_pow2(OC)  # 16
        IC_C = _next_pow2(IC)  # 4
        BLOCK_S = 64

        grid = (N, (spatial_f + BLOCK_S - 1) // BLOCK_S)

        fused_convT_bn_pool_kernel[grid](
            x, weight, scale, shift, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            ODF, OHF, OWF,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            OC_C=OC_C,
            IC_C=IC_C,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )
        return out