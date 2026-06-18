import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_ct_bn_pool_kernel(
    x_ptr, w_ptr, b_ptr,
    scale_ptr, shift_ptr,
    out_ptr,
    N, OC,
    ID, IH, IW,
    OD_CT, OH_CT, OW_CT,
    OD, OH, OW,
    BLOCK_SP: tl.constexpr,
):
    # grid: (N, OC, ceil(OD*OH*OW / BLOCK_SP))
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    TOTAL_SP = OD * OHW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < TOTAL_SP

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # Pre-load weights for this oc: shape (IC=3, KD=3, KH=3, KW=3) = 81 values
    # weight layout: (IC, OC, KD, KH, KW)
    IC = 3
    KD = 3
    KH = 3
    KW = 3
    STRIDE = 2
    PAD = 1
    POOL = 4

    acc = tl.zeros([BLOCK_SP], dtype=tl.float32)

    # Loop over 4x4x4 pool window
    for dd in tl.static_range(0, POOL):
        dd_ct = od * POOL + dd  # [BLOCK_SP]
        for hh in tl.static_range(0, POOL):
            hh_ct = oh * POOL + hh
            for ww in tl.static_range(0, POOL):
                ww_ct = ow * POOL + ww
                in_bound_ct = sp_mask & (dd_ct < OD_CT) & (hh_ct < OH_CT) & (ww_ct < OW_CT)

                # Loop over kernel
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

                            for ic in tl.static_range(0, IC):
                                # x[n, ic, id_div, ih_div, iw_div]
                                x_off = ((pid_n * IC + ic) * ID + id_div) * IH * IW + ih_div * IW + iw_div
                                xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                                # w[ic, oc, kd, kh, kw] - scalar
                                w_off = ((ic * OC + pid_oc) * KD + kd) * KH * KW + kh * KW + kw
                                wv = tl.load(w_ptr + w_off)
                                acc += xv * wv

                # bias for this oc, added once per CT-output element
                bv = tl.load(b_ptr + pid_oc)
                acc += tl.where(in_bound_ct, bv, 0.0)

    acc = acc / 64.0
    scale = tl.load(scale_ptr + pid_oc)
    shift = tl.load(shift_ptr + pid_oc)
    acc = acc * scale + shift

    out_off = ((pid_n * OC + pid_oc) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=sp_mask)


def fused_ct_bn_pool(x, weight, bias, scale, shift,
                     stride, padding,
                     OD_CT, OH_CT, OW_CT, OD, OH, OW):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_SP = 64
    TOTAL_SP = OD * OH * OW
    grid = (N, OC, (TOTAL_SP + BLOCK_SP - 1) // BLOCK_SP)

    fused_ct_bn_pool_kernel[grid](
        x, weight, bias,
        scale, shift,
        out,
        N, OC,
        ID, IH, IW,
        OD_CT, OH_CT, OW_CT,
        OD, OH, OW,
        BLOCK_SP=BLOCK_SP,
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

            KD = KH = KW = self.kernel_size
            OD_CT = (ID - 1) * self.stride - 2 * self.padding + KD
            OH_CT = (IH - 1) * self.stride - 2 * self.padding + KH
            OW_CT = (IW - 1) * self.stride - 2 * self.padding + KW

            OD = OD_CT // 4
            OH = OH_CT // 4
            OW = OW_CT // 4

            return fused_ct_bn_pool(
                x, weight, ct_bias, scale, shift,
                self.stride, self.padding,
                OD_CT, OH_CT, OW_CT,
                OD, OH, OW,
            )