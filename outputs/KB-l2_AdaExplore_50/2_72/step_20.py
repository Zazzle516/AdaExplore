import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose3d + BN(eval) + AvgPool(2) + AvgPool(2) = AvgPool(4)
# Gather formulation: for each output pooled cell (od, oh, ow, oc),
# compute the conv_transpose output for the 4x4x4 sub-window starting at
# (od*4, oh*4, ow*4) and average it, fused with the BN scale/shift epilogue.
#
# ConvTranspose3d output at position (d, h, w) for output channel oc:
#   y[n, oc, d, h, w] = sum_{ic, kd, kh, kw} x[n, ic, di, hi, wi]
#                       * weight[ic, oc, kd, kh, kw]   + bias[oc]
# where di = (d + pad - kd) / stride  (must be integer and in range)
#
# We loop over kd, kh, kw, ic, and the 4x4x4 sub-window.

@triton.jit
def fused_convt_bn_pool_kernel(
    x_ptr,            # [N, IC, ID, IH, IW]
    w_ptr,            # [IC, OC, KD, KH, KW]
    bias_ptr,         # [OC]
    scale_ptr,        # [OC]  (BN folded scale)
    shift_ptr,        # [OC]  (BN folded shift, includes BN affine + conv bias absorption)
    out_ptr,          # [N, OC, OD_OUT, OH_OUT, OW_OUT]
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,       # conv_transpose output spatial (63,63,63)
    OD_OUT, OH_OUT, OW_OUT,  # pooled output (15,15,15)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program ids: (n, pooled_spatial_linear, oc_tile)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)   # linear over OD_OUT*OH_OUT*OW_OUT
    pid_oc = tl.program_id(2)

    ow_out = pid_s % OW_OUT
    tmp = pid_s // OW_OUT
    oh_out = tmp % OH_OUT
    od_out = tmp // OH_OUT

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Starting conv_transpose output coordinates for this 4x4x4 window
    d_start = od_out * 4
    h_start = oh_out * 4
    w_start = ow_out * 4

    # Accumulator: sum over the 4x4x4 window for each oc in tile
    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Loop over the 4x4x4 sub-window
    for dd in tl.static_range(0, 4):
        d = d_start + dd
        for hh in tl.static_range(0, 4):
            h = h_start + hh
            for ww in tl.static_range(0, 4):
                w = w_start + ww
                # in_bounds for conv_transpose output (D=63 etc)
                spatial_ok = (d < OD) & (h < OH) & (w < OW)

                # sum over kernel positions
                # For each (kd, kh, kw): di = (d + PAD - kd), must be divisible by STRIDE, then / STRIDE in [0, ID)
                for kd in tl.static_range(0, KD):
                    d_in_num = d + PAD - kd
                    d_in = d_in_num // STRIDE
                    d_valid = ((d_in_num % STRIDE) == 0) & (d_in >= 0) & (d_in < ID) & spatial_ok

                    for kh in tl.static_range(0, KH):
                        h_in_num = h + PAD - kh
                        h_in = h_in_num // STRIDE
                        h_valid = d_valid & ((h_in_num % STRIDE) == 0) & (h_in >= 0) & (h_in < IH)

                        for kw in tl.static_range(0, KW):
                            w_in_num = w + PAD - kw
                            w_in = w_in_num // STRIDE
                            w_valid = h_valid & ((w_in_num % STRIDE) == 0) & (w_in >= 0) & (w_in < IW)

                            # Reduce over IC
                            for ic in range(0, IC):
                                # x[n, ic, d_in, h_in, w_in]
                                x_off = (pid_n * IC * ID * IH * IW
                                         + ic * ID * IH * IW
                                         + d_in * IH * IW
                                         + h_in * IW
                                         + w_in)
                                xv = tl.load(x_ptr + x_off, mask=w_valid, other=0.0)
                                # weight[ic, oc_tile, kd, kh, kw]
                                w_off = (ic * OC * KD * KH * KW
                                         + oc_offs * KD * KH * KW
                                         + kd * KH * KW
                                         + kh * KW
                                         + kw)
                                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                # If w_valid is False (scalar), contribution is zero via xv=0
                                acc += xv * wv

    # Average over 64 elements (4*4*4)
    acc = acc * (1.0 / 64.0)

    # Apply fused BN scale/shift (which already absorbs conv bias)
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc * scale + shift

    # Store
    out_off = (pid_n * OC * OD_OUT * OH_OUT * OW_OUT
               + oc_offs * OD_OUT * OH_OUT * OW_OUT
               + od_out * OH_OUT * OW_OUT
               + oh_out * OW_OUT
               + ow_out)
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


# Fallback: BN+pool kernel when training or when we want to materialize conv_transpose
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
    ],
    key=['total', 'C'],
)
@triton.jit
def fused_bn_avgpool4_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    total,
    N, C, D, H, W,
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
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

    acc = acc * (1.0 / 64.0)
    acc = acc * scale + shift

    out_ptr_off = (n * out_stride_n + c * out_stride_c +
                   od * out_stride_d + oh * out_stride_h + ow * out_stride_w)
    tl.store(out_ptr + out_ptr_off, acc, mask=mask)


def fused_bn_avgpool4(x, scale, shift):
    N, C, D, H, W = x.shape
    OD = D // 4
    OH = H // 4
    OW = W // 4
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OD * OH * OW
    grid = lambda meta: ((total + meta['BLOCK'] - 1) // meta['BLOCK'],)
    fused_bn_avgpool4_kernel[grid](
        x, out, scale, shift,
        total,
        N, C, D, H, W, OD, OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


def fused_convt_bn_pool(x, weight, bias, scale, shift, stride, padding):
    """
    x: [N, IC, ID, IH, IW]
    weight: [IC, OC, KD, KH, KW]
    bias: [OC]
    scale, shift: [OC] - BN folded with conv bias absorbed into shift
    """
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape

    # Conv transpose output spatial sizes
    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    OD_OUT = OD // 4
    OH_OUT = OH // 4
    OW_OUT = OW // 4

    out = torch.empty((N, OC, OD_OUT, OH_OUT, OW_OUT), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    grid = (N, OD_OUT * OH_OUT * OW_OUT, (OC + BLOCK_OC - 1) // BLOCK_OC)

    fused_convt_bn_pool_kernel[grid](
        x, weight, bias, scale, shift, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        OD_OUT, OH_OUT, OW_OUT,
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride, PAD=padding,
        BLOCK_OC=BLOCK_OC,
        num_warps=4, num_stages=2,
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

    def forward(self, x):
        if self.training:
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = self.avg_pool1(x)
            x = self.avg_pool2(x)
            return x
        else:
            # Fold BN with running stats. Conv bias is already applied by conv_transpose.
            bn = self.batch_norm
            rm = bn.running_mean
            rv = bn.running_var
            eps = bn.eps
            w_bn = bn.weight
            b_bn = bn.bias
            invstd = torch.rsqrt(rv + eps)
            scale = w_bn * invstd  # [OC]
            shift = b_bn - rm * scale  # [OC]

            # Let cuDNN handle conv_transpose (it's already highly optimized)
            x = self.conv_transpose(x)
            return fused_bn_avgpool4(x.contiguous(), scale.contiguous(), shift.contiguous())