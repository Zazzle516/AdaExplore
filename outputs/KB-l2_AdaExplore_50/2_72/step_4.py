import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + BN + 4x AvgPool kernel
# Output shape: (N, OC, OD, OH, OW) where OD = ((D_in*stride + ...)... ) / 4
# We compute via gather: for each output pooled cell, sum over its 4x4x4 region.
# Each region position (od_p, oh_p, ow_p) in conv-transpose output corresponds to:
#   out_pre_bn[n, oc, od_p, oh_p, ow_p] = bias[oc] + sum_{ic, kd, kh, kw} x[n, ic, id, ih, iw] * W[ic, oc, kd, kh, kw]
# where id*stride + kd - pad = od_p, etc., requiring (od_p + pad - kd) % stride == 0
#
# Then pooled = (1/64) * sum_{p in 4x4x4} out_pre_bn[...]
# Final = scale[oc] * pooled + shift[oc]
#
# Reformulate the sum as:
#   pooled = bias[oc] + (1/64) * sum_{p} sum_{ic,kd,kh,kw} x[n,ic,id,ih,iw] * W[ic,oc,kd,kh,kw]
# This still requires the full multiply-add count.

@triton.jit
def fused_convt3d_bn_pool_kernel(
    x_ptr,         # (N, IC, D, H, W)
    w_ptr,         # (IC, OC, KD, KH, KW)
    bias_ptr,      # (OC,)
    scale_ptr,     # (OC,) BN scale
    shift_ptr,     # (OC,) BN shift
    out_ptr,       # (N, OC, OD, OH, OW)
    N, IC, D_in, H_in, W_in,
    OC, KD, KH, KW,
    D_out, H_out, W_out,      # conv-transpose output spatial sizes
    OD, OH, OW,               # pooled output spatial sizes (D_out//4 etc)
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program: (pid_n, pid_spatial, pid_oc)
    pid_n = tl.program_id(0)
    pid_sp = tl.program_id(1)
    pid_oc = tl.program_id(2)

    # decode spatial pid into (od, oh, ow)
    ow = pid_sp % OW
    tmp = pid_sp // OW
    oh = tmp % OH
    od = tmp // OH

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # load bias
    bias = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Accumulator for sum over 4x4x4 region (each is bias + conv contribution)
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # The 4x4x4 pooled region of conv-transpose output positions:
    # od_p in [od*4, od*4+3], similarly for h, w.
    # For each conv-transpose output position, accumulate sum_{ic,kd,kh,kw} x*W

    # Base offsets in conv-transpose output coords
    od_base = od * 4
    oh_base = oh * 4
    ow_base = ow * 4

    # Loop over 64 positions in the pooled window
    for p in tl.static_range(0, 64):
        pd = p // 16
        ph = (p // 4) % 4
        pw = p % 4
        od_p = od_base + pd
        oh_p = oh_base + ph
        ow_p = ow_base + pw

        # Loop over kernel
        for kd in tl.static_range(0, 3):
            id_num = od_p + PAD_D - kd
            id_div = id_num // STRIDE_D
            id_valid = ((id_num % STRIDE_D) == 0) & (id_div >= 0) & (id_div < D_in)
            for kh in tl.static_range(0, 3):
                ih_num = oh_p + PAD_H - kh
                ih_div = ih_num // STRIDE_H
                ih_valid = ((ih_num % STRIDE_H) == 0) & (ih_div >= 0) & (ih_div < H_in)
                for kw in tl.static_range(0, 3):
                    iw_num = ow_p + PAD_W - kw
                    iw_div = iw_num // STRIDE_W
                    iw_valid = ((iw_num % STRIDE_W) == 0) & (iw_div >= 0) & (iw_div < W_in)
                    valid = id_valid & ih_valid & iw_valid

                    if valid:
                        # Sum over IC
                        # x[n, ic, id_div, ih_div, iw_div] * W[ic, oc, kd, kh, kw]
                        x_base = pid_n * (IC * D_in * H_in * W_in) + id_div * (H_in * W_in) + ih_div * W_in + iw_div
                        w_base = kd * (KH * KW) + kh * KW + kw  # offset within (kd,kh,kw)

                        for ic in tl.static_range(0, 3):  # IC=3
                            x_val = tl.load(x_ptr + x_base + ic * (D_in * H_in * W_in))
                            # W shape: (IC, OC, KD, KH, KW) -> stride for oc = KD*KH*KW
                            w_ptr_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_base
                            w_val = tl.load(w_ptr + w_ptr_off, mask=oc_mask, other=0.0)
                            acc += x_val * w_val

        # bias contribution per position
        acc += bias

    pooled = acc / 64.0
    result = pooled * scale + shift

    # Store
    out_off = pid_n * (OC * OD * OH * OW) + oc_offs * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow
    tl.store(out_ptr + out_off, result, mask=oc_mask)


def fused_convt3d_bn_pool(x, weight, bias, scale, shift, stride, padding):
    N, IC, D_in, H_in, W_in = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    # ConvTranspose3d output size
    D_out = (D_in - 1) * stride - 2 * padding + KD
    H_out = (H_in - 1) * stride - 2 * padding + KH
    W_out = (W_in - 1) * stride - 2 * padding + KW

    OD = D_out // 4
    OH = H_out // 4
    OW = W_out // 4

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    BLOCK_OC = 16  # OC=16, exactly one block
    grid = (N, OD * OH * OW, (OC + BLOCK_OC - 1) // BLOCK_OC)

    fused_convt3d_bn_pool_kernel[grid](
        x, weight, bias, scale, shift, out,
        N, IC, D_in, H_in, W_in,
        OC, KD, KH, KW,
        D_out, H_out, W_out,
        OD, OH, OW,
        stride, stride, stride,
        padding, padding, padding,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
        num_stages=2,
    )
    return out


# Fallback: just BN+pool fused (when training-mode BN needed we just call torch ops)
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
            # Fold BN
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

            # Check supported shapes
            IC = weight.shape[0]
            OC = weight.shape[1]
            KD, KH, KW = weight.shape[2], weight.shape[3], weight.shape[4]

            if IC == 3 and KD == 3 and KH == 3 and KW == 3 and self.stride == 2 and self.padding == 1:
                return fused_convt3d_bn_pool(
                    x, weight, conv_bias, scale.contiguous(), shift.contiguous(),
                    self.stride, self.padding,
                )
            else:
                x = self.conv_transpose(x)
                return fused_bn_avgpool4(x.contiguous(), scale.contiguous(), shift.contiguous())