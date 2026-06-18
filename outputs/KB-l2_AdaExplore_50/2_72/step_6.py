import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-style ConvTranspose3d fused with BN + 4x4x4 AvgPool.
# For each input voxel (n, ic, id, ih, iw) and each kernel position (kd, kh, kw),
# the contribution lands at conv-transpose output position:
#   od_p = id*stride + kd - pad
#   oh_p = ih*stride + kh - pad
#   ow_p = iw*stride + kw - pad
# which is in pooled output cell (od_p//4, oh_p//4, ow_p//4) (if within bounds).
# We accumulate via atomic_add into the pooled output buffer scaled by 1/64.
# Then a small elementwise kernel adds bias*(window_count/64) and applies BN scale/shift.
#
# But: bias contribution per pooled output is bias[oc] (sum over 64 positions of bias/64 = bias).
# However only valid output positions contribute; since stride=2, pad=1, kernel=3, output is dense.
# D_out = (32-1)*2 - 2 + 3 = 63, so 4x4x4 windows: 63//4 = 15 pooled cells along each dim,
# but conv-transpose output is 63 -> avg_pool kernel=2 -> 31 -> avg_pool kernel=2 -> 15.
# Wait, AvgPool3d with kernel_size=2 default stride=2, so 63->31 (floor), 31->15. 
# So pooled output is 15x15x15, and each pooled cell covers 4x4x4 of conv-transpose output... 
# but not exactly because of the two-step pooling with odd intermediate sizes.
#
# Actually: AvgPool with kernel=2,stride=2 on size 63 gives floor(63/2)=31, dropping last element.
# Then 31 -> floor(31/2) = 15, dropping last. So the pooled cell (od,oh,ow) corresponds to
# conv-transpose positions od*4..od*4+3 (and similarly), all within [0, 60+3=63) which is valid.
# Yes, all 4x4x4 = 64 positions fit. Good.


@triton.jit
def scatter_convt_pool_kernel(
    x_ptr,         # (N, IC, D_in, H_in, W_in)
    w_ptr,         # (IC, OC, KD, KH, KW)
    out_ptr,       # (N, OC, OD, OH, OW) - accumulator (pre-bias, pre-BN)
    N, IC, D_in, H_in, W_in,
    OC,
    OD, OH, OW,
    STRIDE: tl.constexpr,
    PAD: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program: (pid_input_voxel, pid_oc_block)
    # pid_input_voxel encodes (n, id, ih, iw); we loop over IC inside.
    pid_v = tl.program_id(0)
    pid_oc = tl.program_id(1)

    iw = pid_v % W_in
    tmp = pid_v // W_in
    ih = tmp % H_in
    tmp = tmp // H_in
    id_ = tmp % D_in
    n = tmp // D_in

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Preload x[n, :, id, ih, iw] for all IC (small, IC=3)
    x_base = n * (IC * D_in * H_in * W_in) + id_ * (H_in * W_in) + ih * W_in + iw
    # We'll iterate over kd, kh, kw and IC.

    for kd in tl.static_range(0, KD):
        od_p = id_ * STRIDE + kd - PAD
        for kh in tl.static_range(0, KH):
            oh_p = ih * STRIDE + kh - PAD
            for kw in tl.static_range(0, KW):
                ow_p = iw * STRIDE + kw - PAD
                # Check bounds: must be within conv-transpose output [0, D_out)
                # D_out = (D_in-1)*STRIDE - 2*PAD + KD
                # But we only care about pooled cells in [0, OD), so divide by 4.
                # Pooled cell = od_p // 4 (only valid if od_p >= 0 and od_p < OD*4)
                pod = od_p // 4
                poh = oh_p // 4
                pow_ = ow_p // 4
                valid = (od_p >= 0) & (oh_p >= 0) & (ow_p >= 0) & \
                        (pod < OD) & (poh < OH) & (pow_ < OW)

                if valid:
                    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)
                    # Loop over IC, accumulate x * W into oc vector
                    for ic in tl.static_range(0, 3):  # IC=3
                        x_val = tl.load(x_ptr + x_base + ic * (D_in * H_in * W_in))
                        # W[ic, oc, kd, kh, kw]
                        w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + \
                                kd * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        acc += x_val * w_val

                    # Scatter into output buffer
                    out_off = n * (OC * OD * OH * OW) + oc_offs * (OD * OH * OW) + \
                              pod * (OH * OW) + poh * OW + pow_
                    tl.atomic_add(out_ptr + out_off, acc, mask=oc_mask)


@triton.jit
def bn_bias_finalize_kernel(
    inout_ptr,     # (N, OC, OD, OH, OW)
    bias_ptr,      # (OC,)
    scale_ptr,     # (OC,)
    shift_ptr,     # (OC,)
    total, OC, ODOHW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Decode channel index
    c = (offs // ODOHW) % OC
    val = tl.load(inout_ptr + offs, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)
    # val currently = sum over 64 positions of conv contributions (no bias)
    # pooled = (val + 64*bias) / 64 = val/64 + bias
    # final = scale * pooled + shift
    pooled = val * (1.0 / 64.0) + bias
    out = scale * pooled + shift
    tl.store(inout_ptr + offs, out, mask=mask)


def fused_scatter_convt_bn_pool(x, weight, conv_bias, scale, shift, stride, padding):
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape

    D_out = (D_in - 1) * stride - 2 * padding + KD
    H_out = (H_in - 1) * stride - 2 * padding + KH
    W_out = (W_in - 1) * stride - 2 * padding + KW

    OD = D_out // 4
    OH = H_out // 4
    OW = W_out // 4

    # zero-initialized accumulator
    out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 16
    grid = (N * D_in * H_in * W_in, (OC + BLOCK_OC - 1) // BLOCK_OC)

    scatter_convt_pool_kernel[grid](
        x, weight, out,
        N, IC, D_in, H_in, W_in,
        OC, OD, OH, OW,
        STRIDE=stride, PAD=padding,
        KD=KD, KH=KH, KW=KW,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
        num_stages=2,
    )

    total = N * OC * OD * OH * OW
    ODOHW = OD * OH * OW
    BLOCK = 256
    grid2 = ((total + BLOCK - 1) // BLOCK,)
    bn_bias_finalize_kernel[grid2](
        out, conv_bias, scale, shift,
        total, OC, ODOHW,
        BLOCK=BLOCK, num_warps=4,
    )
    return out


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
            scale = (bn.weight * invstd).contiguous()
            shift = (bn.bias - bn.running_mean * bn.weight * invstd).contiguous()

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
                return fused_scatter_convt_bn_pool(
                    x, weight, conv_bias, scale, shift, self.stride, self.padding,
                )
            else:
                x = self.conv_transpose(x)
                return fused_bn_avgpool4(x.contiguous(), scale, shift)