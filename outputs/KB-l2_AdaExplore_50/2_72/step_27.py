import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Gather-based fused ConvTranspose3d + BN + 4x4x4 AvgPool.
# For pooled output cell (n, oc, od, oh, ow):
#   pooled = scale[oc] * (bias[oc] + (1/64) * sum_{p in 4x4x4 window} conv_out[p]) + shift[oc]
# where conv_out[p] = sum_{ic, kd, kh, kw with valid mapping} x[n, ic, id, ih, iw] * W[ic, oc, kd, kh, kw]
# and stride=2, pad=1, k=3 -> id = (od_p + 1 - kd) / 2, valid when (od_p + 1 - kd) is even.
#
# Strategy:
# - One program per (n, pooled_spatial_idx). OC=16 handled as a vector inside the kernel.
# - Iterate over 4x4x4 = 64 positions; for each, iterate kd,kh,kw (3^3 = 27) checking validity;
#   for each valid (kd,kh,kw), load 3 input voxels (IC=3) and 3*16 weights.
# - Note: For each conv-transpose output position only positions where (od_p+1-kd) is even contribute,
#   so roughly ~8 valid (kd,kh,kw) per output position.
#
# Reorganization: instead of looping 64 * 27, recognize that the valid kernel taps for each od_p depend
# on od_p%2. We can iterate over input voxels (id, ih, iw) that touch the pooled window. The pooled
# window covers od_p in [od*4, od*4+3], i.e., 4 consecutive output positions. Input id contributing is
# id = (od_p + 1 - kd) / 2 with kd in {0,1,2}. The range of id for od_p in [od*4, od*4+3] and kd in 0..2
# is id in [(od*4-1)/2, (od*4+3+1)/2] => id roughly in [od*2, od*2+2] (3 values).
# So each pooled cell sees a 3x3x3 grid of input voxels (in worst case).
#
# Better: loop over (di, dh, dw) in {0,1,2} -> id = od*2 + di - 1+? Actually with pad=1 and stride=2:
# For output position p in [4od, 4od+3], we need id*2 - 1 + kd = p, so id = (p + 1 - kd)/2.
# Let id = od*2 + a, where a in {-1, 0, 1, 2}? Let's just enumerate input positions and check which
# pooled-window outputs they touch.
#
# Simplest correct + fast: per pooled cell, loop over 3 input positions in each spatial dim (9 input voxels in d×h×w = 27 input voxels), then for each input voxel compute all kernel taps that map into the pooled window (each input voxel contributes to multiple output positions). Actually let's just stick with: loop over the 27 (kd, kh, kw) taps explicitly and over the 64 (pd, ph, pw) positions, doing static checks — but that's 64*27 iters which is a lot.
#
# Cleaner: per pooled cell, loop over input voxels in the receptive field. The receptive field of a
# 4x4x4 window of output (each output uses 3x3x3 input neighborhood at stride 2) is at most
# ceil((4 + 2)/2) = 3 input voxels per dim (since adjacent outputs share inputs).
# Actually: outputs od_p in [4od, 4od+3], inputs id in {(od_p+1-kd)/2}. The union of all valid id
# values is 3 per dim. So 3^3 = 27 input voxels per pooled cell.
# For each input voxel (id, ih, iw), sum contributions from all (kd, kh, kw) such that
# (id*2 - 1 + kd) is in [4od, 4od+3], etc.
# So loop 27 input voxels * up to 27 kernel taps (but constrained), but actually for each input voxel,
# the kernel taps that map into the pooled window are determined: each input voxel hits exactly the
# (kd, kh, kw) for which the resulting output is in [4od, 4od+3]^3, i.e., effectively all kernel taps
# (since the input voxel can produce up to 3 output positions per dim from kd∈{0,1,2}).

@triton.jit
def fused_gather_convt_bn_pool_kernel(
    x_ptr,         # (N, IC, D_in, H_in, W_in)
    w_ptr,         # (IC, OC, KD, KH, KW)
    bias_ptr,      # (OC,)
    scale_ptr,     # (OC,)
    shift_ptr,     # (OC,)
    out_ptr,       # (N, OC, OD, OH, OW)
    N, IC, D_in, H_in, W_in,
    OC, D_out, H_out, W_out,
    OD, OH, OW,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)

    ow = pid_sp % OW
    tmp = pid_sp // OW
    oh = tmp % OH
    od = tmp // OH

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Output window: od_p in [4*od, 4*od+3], same for h, w.
    # Need id*2 - 1 + kd in [4od, 4od+3] => id in [(4od + 1 - kd)/2, (4od + 4 - kd)/2]
    # Across kd in {0,1,2}, the union of id values is {2od-1, 2od, 2od+1, 2od+2} but the 4 values
    # depend on parity. Use static range over 4 candidate input voxels per dim (covering all cases)
    # and test validity per (kd, kh, kw).
    #
    # Simpler & still correct: loop over (kd, kh, kw) in 3^3, and for each, loop over (pd, ph, pw)
    # in 4^3 (64 outputs) checking validity. This is 27*64 = 1728 iters per (n, pooled_voxel).
    # That's too many. Instead: loop over input voxel candidates in 3^3, and for each, accumulate
    # the contribution to all 4 output positions per dim that it touches.
    #
    # Best plan: iterate over (kd, kh, kw) in static_range(3) and (pd, ph, pw) in static_range(4).
    # For each combo, derive id, ih, iw and validity. With stride=2, pad=1:
    #   id = (4*od + pd + 1 - kd) / 2, valid when (4*od + pd + 1 - kd) even and in [0, D_in)
    # That's 27 * 64 = 1728 static iterations. Likely manageable since each does 3 loads + FMA.
    #
    # Optimization: rewrite as static range over a = (pd + 1 - kd), b similarly. For each kd, the
    # parity of (pd+1-kd) is what matters; only even values give valid id. So per kd: pd in {kd-1, kd+1}
    # roughly. Let's just do the straightforward 27*64 loop.

    D_in_HW = D_in * H_in * W_in
    HW_in = H_in * W_in
    OC_KDHW = OC * 27
    KDHW = 27

    for kd in tl.static_range(0, 3):
        for kh in tl.static_range(0, 3):
            for kw in tl.static_range(0, 3):
                k_off = kd * 9 + kh * 3 + kw
                # Load weights for all OC: W[ic, oc, kd, kh, kw], ic in {0,1,2}
                w0 = tl.load(w_ptr + 0 * OC_KDHW + oc_offs * KDHW + k_off, mask=oc_mask, other=0.0)
                w1 = tl.load(w_ptr + 1 * OC_KDHW + oc_offs * KDHW + k_off, mask=oc_mask, other=0.0)
                w2 = tl.load(w_ptr + 2 * OC_KDHW + oc_offs * KDHW + k_off, mask=oc_mask, other=0.0)

                for pd in tl.static_range(0, 4):
                    d_num = 4 * od + pd + 1 - kd
                    # d_num must be >= 0, even, and d_num/2 < D_in
                    d_par_ok = (d_num & 1) == 0
                    id_ = d_num // 2  # floor; but if d_num<0, floor differs. We mask via range check.
                    d_range_ok = (d_num >= 0) & (id_ < D_in)
                    d_ok = d_par_ok & d_range_ok

                    for ph in tl.static_range(0, 4):
                        h_num = 4 * oh + ph + 1 - kh
                        h_par_ok = (h_num & 1) == 0
                        ih_ = h_num // 2
                        h_range_ok = (h_num >= 0) & (ih_ < H_in)
                        h_ok = h_par_ok & h_range_ok

                        for pw in tl.static_range(0, 4):
                            w_num = 4 * ow + pw + 1 - kw
                            w_par_ok = (w_num & 1) == 0
                            iw_ = w_num // 2
                            w_range_ok = (w_num >= 0) & (iw_ < W_in)
                            valid = d_ok & h_ok & w_ok & w_par_ok & w_range_ok

                            x_off = pid_n * (IC * D_in_HW) + id_ * HW_in + ih_ * W_in + iw_
                            x0 = tl.load(x_ptr + x_off + 0 * D_in_HW, mask=valid, other=0.0)
                            x1 = tl.load(x_ptr + x_off + 1 * D_in_HW, mask=valid, other=0.0)
                            x2 = tl.load(x_ptr + x_off + 2 * D_in_HW, mask=valid, other=0.0)

                            acc += x0 * w0 + x1 * w1 + x2 * w2

    # 64 positions: bias contribution per pooled cell = bias * 64 / 64 = bias (all positions valid for OD=15, D_out=63? D_out=63, OD=15 -> covers 0..59, all valid)
    pooled = acc * (1.0 / 64.0) + bias
    result = pooled * scale + shift

    out_off = pid_n * (OC * OD * OH * OW) + oc_offs * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptr + out_off, result, mask=oc_mask)


def fused_gather_convt_bn_pool(x, weight, conv_bias, scale, shift, stride, padding):
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w and stride == 2 and padding == 1 and KD == 3 and KH == 3 and KW == 3

    D_out = (D_in - 1) * stride - 2 * padding + KD
    H_out = (H_in - 1) * stride - 2 * padding + KH
    W_out = (W_in - 1) * stride - 2 * padding + KW

    OD = D_out // 4
    OH = H_out // 4
    OW = W_out // 4

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 16
    grid = (N, OD * OH * OW)
    fused_gather_convt_bn_pool_kernel[grid](
        x, weight, conv_bias, scale, shift, out,
        N, IC, D_in, H_in, W_in,
        OC, D_out, H_out, W_out,
        OD, OH, OW,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
        num_stages=2,
    )
    return out


# Fallback BN+pool kernel
@triton.jit
def fused_bn_avgpool4_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    d0 = od * 4
    h0 = oh * 4
    w0 = ow * 4

    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)

    acc = tl.zeros([BLOCK], dtype=tl.float32)
    base = n * (C * D * H * W) + c * (D * H * W)

    for dd in tl.static_range(0, 4):
        for hh in tl.static_range(0, 4):
            for ww in tl.static_range(0, 4):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                in_bounds = mask & (d_idx < D) & (h_idx < H) & (w_idx < W)
                ptr = base + d_idx * (H * W) + h_idx * W + w_idx
                v = tl.load(x_ptr + ptr, mask=in_bounds, other=0.0)
                acc += v

    acc = acc / 64.0
    acc = acc * scale + shift

    out_off = n * (C * OD * OH * OW) + c * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=mask)


def fused_bn_avgpool4(x, scale, shift):
    N, C, D, H, W = x.shape
    OD = D // 4
    OH = H // 4
    OW = W // 4
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OD * OH * OW
    BLOCK = 256
    grid = ((total + BLOCK - 1) // BLOCK,)
    fused_bn_avgpool4_kernel[grid](
        x, out, scale, shift,
        N, C, D, H, W, OD, OH, OW,
        BLOCK=BLOCK, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.avg_pool1(x)
            x = self.avg_pool2(x)
            return x
        else:
            bn = self.batch_norm
            invstd = torch.rsqrt(bn.running_var + bn.eps)
            scale = bn.weight * invstd
            shift = bn.bias - bn.running_mean * scale

            x = x.contiguous()
            weight = self.conv_transpose.weight.contiguous()
            conv_bias = self.conv_transpose.bias
            if conv_bias is None:
                conv_bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
            else:
                conv_bias = conv_bias.contiguous()

            IC = weight.shape[0]
            OC = weight.shape[1]
            KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]

            if IC == 3 and KD == 3 and KH == 3 and KW == 3 and self.stride == 2 and self.padding == 1:
                return fused_gather_convt_bn_pool(
                    x, weight, conv_bias.contiguous(),
                    scale.contiguous(), shift.contiguous(),
                    self.stride, self.padding,
                )
            else:
                x = self.conv_transpose(x)
                return fused_bn_avgpool4(x.contiguous(), scale.contiguous(), shift.contiguous())