import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + BN + AvgPool(4x4x4) kernel
# One program per (N, OC_block, OD_tile, OH_tile, OW_tile) producing pooled output
# We exploit:
#   - IC = 3, OC = 16, KD=KH=KW=3, stride=2, pad=1
#   - Pre-pool output size = ID*2 (since (ID-1)*2 - 2 + 3 = 2*ID - 1... )
#     Actually: (32-1)*2 - 2*1 + 3 = 62 - 2 + 3 = 63 — not divisible by 4!
# So PD_OD = 63 in this config. Hmm let me re-check.
# Actually output of nn.ConvTranspose3d with stride=2,padding=1,kernel=3 from 32:
#   out = (32-1)*2 - 2*1 + 3 + output_padding(0) = 63
# But pool 2x then pool 2x => 63//2 = 31, 31//2 = 15 (floor)
# Reference kernels assume PD_*//4 — but 63//4 = 15, with last row truncated.
# Actually torch AvgPool3d with kernel=2 stride=2 on 63 gives 31 (floor), then 15.
# Same as 60//4 = 15. So effectively we use 60 of 63. Let's handle generally.
#
# We'll compute pooled output for OD = floor(floor(PD/2)/2) and only use
# pre-pool positions in [0, 4*OD). This matches torch behavior (default avgpool).

@triton.jit
def fused_convt_bn_pool_kernel(
    x_ptr,            # input: (N, IC, ID, IH, IW)
    w_ptr,            # weight: (IC, OC, KD, KH, KW)
    b_ptr,            # conv bias: (OC,)
    scale_ptr,        # BN scale (OC,)
    shift_ptr,        # BN shift (OC,)
    out_ptr,          # output: (N, OC, OD, OH, OW) -- pooled
    N, IC, OC,
    ID, IH, IW,
    PD_OD, PD_OH, PD_OW,
    OD, OH, OW,
    stride_p: tl.constexpr,
    pad: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # pre-pool window start
    pd_d0 = od * 4
    pd_h0 = oh * 4
    pd_w0 = ow * 4

    # Load bias once
    bv = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Iterate over 4x4x4 = 64 pre-pool output positions
    for dd in tl.static_range(0, 4):
        pd_d = pd_d0 + dd
        for hh in tl.static_range(0, 4):
            pd_h = pd_h0 + hh
            for ww in tl.static_range(0, 4):
                pd_w = pd_w0 + ww
                voxel = tl.zeros([BLOCK_OC], dtype=tl.float32)

                for kd in tl.static_range(0, 3):
                    id_num = pd_d + pad - kd
                    id_q = id_num // stride_p
                    id_ok = (id_num >= 0) & (id_num - id_q * stride_p == 0) & (id_q >= 0) & (id_q < ID)
                    for kh in tl.static_range(0, 3):
                        ih_num = pd_h + pad - kh
                        ih_q = ih_num // stride_p
                        ih_ok = (ih_num >= 0) & (ih_num - ih_q * stride_p == 0) & (ih_q >= 0) & (ih_q < IH)
                        for kw in tl.static_range(0, 3):
                            iw_num = pd_w + pad - kw
                            iw_q = iw_num // stride_p
                            iw_ok = (iw_num >= 0) & (iw_num - iw_q * stride_p == 0) & (iw_q >= 0) & (iw_q < IW)
                            valid = id_ok & ih_ok & iw_ok
                            if valid:
                                x_base = (n * IC * ID * IH * IW
                                          + id_q * IH * IW
                                          + ih_q * IW
                                          + iw_q)
                                w_base = (oc_offs * 27
                                          + kd * 9
                                          + kh * 3
                                          + kw)
                                # unroll IC=3
                                x0 = tl.load(x_ptr + x_base + 0 * ID * IH * IW)
                                w0 = tl.load(w_ptr + w_base + 0 * OC * 27, mask=oc_mask, other=0.0)
                                voxel += x0 * w0
                                x1 = tl.load(x_ptr + x_base + 1 * ID * IH * IW)
                                w1 = tl.load(w_ptr + w_base + 1 * OC * 27, mask=oc_mask, other=0.0)
                                voxel += x1 * w1
                                x2 = tl.load(x_ptr + x_base + 2 * ID * IH * IW)
                                w2 = tl.load(w_ptr + w_base + 2 * OC * 27, mask=oc_mask, other=0.0)
                                voxel += x2 * w2
                voxel += bv
                acc += voxel

    acc = acc / 64.0
    acc = acc * scale + shift

    out_off = (n * OC * OD * OH * OW
               + oc_offs * OD * OH * OW
               + od * OH * OW
               + oh * OW
               + ow)
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


def fused_convt_bn_pool(x, weight, bias, scale, shift, stride, pad):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    PD_OD = (ID - 1) * stride - 2 * pad + KD
    PD_OH = (IH - 1) * stride - 2 * pad + KH
    PD_OW = (IW - 1) * stride - 2 * pad + KW
    OD = (PD_OD // 2) // 2
    OH = (PD_OH // 2) // 2
    OW = (PD_OW // 2) // 2

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    grid = (N * OD * OH * OW,)

    fused_convt_bn_pool_kernel[grid](
        x, weight, bias, scale, shift, out,
        N, IC, OC,
        ID, IH, IW,
        PD_OD, PD_OH, PD_OW,
        OD, OH, OW,
        stride, pad,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
        num_stages=2,
    )
    return out


@triton.jit
def fused_bn_avgpool4_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
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
    base = n * stride_n + c * stride_c

    for dd in tl.static_range(0, 4):
        for hh in tl.static_range(0, 4):
            for ww in tl.static_range(0, 4):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                in_bounds = mask & (d_idx < D) & (h_idx < H) & (w_idx < W)
                ptr = base + d_idx * stride_d + h_idx * stride_h + w_idx * stride_w
                v = tl.load(x_ptr + ptr, mask=in_bounds, other=0.0)
                acc += v

    acc = acc / 64.0
    acc = acc * scale + shift

    out_off = (n * out_stride_n + c * out_stride_c +
               od * out_stride_d + oh * out_stride_h + ow * out_stride_w)
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
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
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
        self._stride = stride
        self._padding = padding
        self._kernel_size = kernel_size

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.avg_pool1(x)
            x = self.avg_pool2(x)
            return x
        else:
            bn = self.batch_norm
            rm = bn.running_mean
            rv = bn.running_var
            eps = bn.eps
            w = bn.weight
            b = bn.bias
            invstd = torch.rsqrt(rv + eps)
            scale = (w * invstd).contiguous()
            shift = (b - rm * scale).contiguous()

            weight = self.conv_transpose.weight  # (IC, OC, KD, KH, KW)
            cb = self.conv_transpose.bias
            if cb is None:
                conv_bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
            else:
                conv_bias = cb.contiguous()

            N, IC, ID, IH, IW = x.shape
            _, OC, KD, KH, KW = weight.shape

            x = x.contiguous()
            weight_c = weight.contiguous()

            if (IC == 3 and OC == 16 and KD == 3 and KH == 3 and KW == 3
                    and self._stride == 2 and self._padding == 1):
                return fused_convt_bn_pool(
                    x, weight_c, conv_bias, scale, shift,
                    self._stride, self._padding,
                )

            y = self.conv_transpose(x)
            return fused_bn_avgpool4(y.contiguous(), scale, shift)