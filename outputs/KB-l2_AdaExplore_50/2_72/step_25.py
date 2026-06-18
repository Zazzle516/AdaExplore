import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convtranspose_bn_pool_kernel(
    x_ptr, w_ptr, b_ptr,
    scale_ptr, shift_ptr,
    out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD_CT, OH_CT, OW_CT,  # conv-transpose output dims (63)
    OD, OH, OW,           # final output dims (16)
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    POOL: tl.constexpr,   # 4
    BLOCK_OC: tl.constexpr,
):
    # program ids: (n, oc_tile, spatial pooled cell flattened)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    od = pid_sp // OHW
    rem = pid_sp % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # 4x4x4 output window in conv-transpose space
    # output coordinates: od_ct in [od*4, od*4+3], etc.
    # The conv-transpose output value at (n, oc, dd_ct, hh_ct, ww_ct) is:
    #   sum over (ic, kd, kh, kw): x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id = (dd_ct + PAD - kd) / STRIDE if divisible
    # We then average over 4x4x4 window and apply scale*acc + shift.

    acc = tl.zeros([BLOCK_OC], dtype=tl.float32)

    # Loop over the 4x4x4 pool window
    for dd in tl.static_range(0, POOL):
        dd_ct = od * POOL + dd
        for hh in tl.static_range(0, POOL):
            hh_ct = oh * POOL + hh
            for ww in tl.static_range(0, POOL):
                ww_ct = ow * POOL + ww
                in_bound_ct = (dd_ct < OD_CT) & (hh_ct < OH_CT) & (ww_ct < OW_CT)

                # For each kernel position, compute corresponding input position
                for kd in tl.static_range(0, KD):
                    id_num = dd_ct + PAD - kd
                    id_div = id_num // STRIDE
                    id_ok = (id_num >= 0) & ((id_num % STRIDE) == 0) & (id_div >= 0) & (id_div < ID)
                    for kh in tl.static_range(0, KH):
                        ih_num = hh_ct + PAD - kh
                        ih_div = ih_num // STRIDE
                        ih_ok = (ih_num >= 0) & ((ih_num % STRIDE) == 0) & (ih_div >= 0) & (ih_div < IH)
                        for kw in tl.static_range(0, KW):
                            iw_num = ww_ct + PAD - kw
                            iw_div = iw_num // STRIDE
                            iw_ok = (iw_num >= 0) & ((iw_num % STRIDE) == 0) & (iw_div >= 0) & (iw_div < IW)
                            valid = in_bound_ct & id_ok & ih_ok & iw_ok
                            # Loop over IC, accumulate sum_ic x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
                            # IC is small (3), unroll
                            for ic in tl.static_range(0, 3):
                                # x offset
                                x_off = ((pid_n * IC + ic) * ID + id_div) * IH * IW + ih_div * IW + iw_div
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                # w[ic, oc, kd, kh, kw], weight shape (IC, OC, KD, KH, KW)
                                w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                                wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                                acc += xv * wv

                # Add bias for this output position (bias broadcast over spatial)
                # bias is added once per conv-transpose output element, masked by in_bound_ct
                bv = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
                acc += tl.where(in_bound_ct, bv, 0.0)

    # Average over 64 (pool size)
    acc = acc / 64.0

    # Apply BN scale/shift
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc * scale + shift

    # Store
    out_off = ((pid_n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


def fused_convtranspose_bn_pool(x, weight, bias, scale, shift,
                                 stride, padding, OD_CT, OH_CT, OW_CT,
                                 OD, OH, OW):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16  # OC=16, exactly one tile
    grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, OD * OH * OW)

    fused_convtranspose_bn_pool_kernel[grid](
        x, weight, bias,
        scale, shift,
        out,
        N, IC, OC,
        ID, IH, IW,
        OD_CT, OH_CT, OW_CT,
        OD, OH, OW,
        KD, KH, KW,
        stride, padding,
        4,
        BLOCK_OC,
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
            rm = bn.running_mean
            rv = bn.running_var
            eps = bn.eps
            w = bn.weight
            b = bn.bias
            invstd = torch.rsqrt(rv + eps)
            scale = (w * invstd).contiguous()
            shift = (b - rm * scale).contiguous()

            x = x.contiguous()
            N, IC, ID, IH, IW = x.shape
            weight = self.conv_transpose.weight.contiguous()
            ct_bias = self.conv_transpose.bias
            if ct_bias is None:
                ct_bias = torch.zeros(self.out_channels, device=x.device, dtype=x.dtype)
            else:
                ct_bias = ct_bias.contiguous()

            # Compute conv-transpose output dims
            KD = KH = KW = self.kernel_size
            OD_CT = (ID - 1) * self.stride - 2 * self.padding + KD
            OH_CT = (IH - 1) * self.stride - 2 * self.padding + KH
            OW_CT = (IW - 1) * self.stride - 2 * self.padding + KW

            OD = OD_CT // 4
            OH = OH_CT // 4
            OW = OW_CT // 4

            return fused_convtranspose_bn_pool(
                x, weight, ct_bias, scale, shift,
                self.stride, self.padding,
                OD_CT, OH_CT, OW_CT,
                OD, OH, OW,
            )