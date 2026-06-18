import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    PD, PH, PW,  # pooled dims
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
):
    # one program per (n, oc, pd, ph, pw)
    pid = tl.program_id(0)
    pw = pid % PW
    pid1 = pid // PW
    ph = pid1 % PH
    pid2 = pid1 // PH
    pd = pid2 % PD
    pid3 = pid2 // PD
    oc = pid3 % OC
    n = pid3 // OC

    # the output region this pool window covers:
    # pool is 2x2x2 stride 2 padding 0, so the 8 output positions are
    # od in {pd*2, pd*2+1}, etc.
    od0 = pd * 2
    oh0 = ph * 2
    ow0 = pw * 2

    # We'll compute the 8 output values (od0..od0+1, oh0..oh0+1, ow0..ow0+1)
    # then take the max.

    # bias
    bias = tl.load(b_ptr + oc).to(tl.float32)

    max_val = -float('inf')

    # iterate over the 8 spatial positions
    for ddi in tl.static_range(0, 2):
        for hhi in tl.static_range(0, 2):
            for wwi in tl.static_range(0, 2):
                od = od0 + ddi
                oh = oh0 + hhi
                ow = ow0 + wwi

                acc = bias

                # ConvTranspose3d output:
                # out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw}
                #   input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
                # where id*SD - PAD_D + kd = od => id = (od + PAD_D - kd) / SD
                # only valid when (od + PAD_D - kd) % SD == 0

                for kd in tl.static_range(0, KD):
                    id_num = od + PAD_D - kd
                    id_val = id_num // SD
                    id_valid = ((id_num - id_val * SD) == 0) & (id_val >= 0) & (id_val < ID)

                    for kh in tl.static_range(0, KH):
                        ih_num = oh + PAD_H - kh
                        ih_val = ih_num // SH
                        ih_valid = ((ih_num - ih_val * SH) == 0) & (ih_val >= 0) & (ih_val < IH)

                        for kw in tl.static_range(0, KW):
                            iw_num = ow + PAD_W - kw
                            iw_val = iw_num // SW
                            iw_valid = ((iw_num - iw_val * SW) == 0) & (iw_val >= 0) & (iw_val < IW)

                            valid = id_valid & ih_valid & iw_valid

                            # gather across IC
                            ic_offs = tl.arange(0, IC_C)
                            ic_mask = ic_offs < IC

                            # input[n, ic, id, ih, iw]
                            x_off = (((n * IC + ic_offs) * ID + id_val) * IH + ih_val) * IW + iw_val
                            xv = tl.load(x_ptr + x_off, mask=ic_mask & valid, other=0.0).to(tl.float32)

                            # weight[ic, oc, kd, kh, kw]
                            w_off = (((ic_offs * OC + oc) * KD + kd) * KH + kh) * KW + kw
                            wv = tl.load(w_ptr + w_off, mask=ic_mask, other=0.0).to(tl.float32)

                            acc += tl.sum(xv * wv, axis=0)

                if acc > max_val:
                    max_val = acc

    # store to out[n, oc, pd, ph, pw]
    out_off = (((n * OC + oc) * PD + pd) * PH + ph) * PW + pw
    tl.store(out_ptr + out_off, max_val)


def fused_conv_transpose_maxpool(x, weight, bias,
                                  stride, padding, output_padding,
                                  pool_kernel, pool_stride, pool_padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride, stride, stride
    PAD_D, PAD_H, PAD_W = padding, padding, padding
    OPD, OPH, OPW = output_padding, output_padding, output_padding

    OD = (ID - 1) * SD - 2 * PAD_D + KD + OPD
    OH = (IH - 1) * SH - 2 * PAD_H + KH + OPH
    OW = (IW - 1) * SW - 2 * PAD_W + KW + OPW

    # pool: 2x2x2 stride 2 padding 0
    PD = OD // 2
    PH = OH // 2
    PW = OW // 2

    out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=torch.float32)

    IC_C = triton.next_power_of_2(IC)
    OC_C = triton.next_power_of_2(OC)

    grid = (N * OC * PD * PH * PW,)
    conv_transpose_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        PD, PH, PW,
        KD, KH, KW,
        SD, SH, SW,
        PAD_D, PAD_H, PAD_W,
        IC_C=IC_C,
        OC_C=OC_C,
        num_warps=1,
    )
    return out


@triton.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // S
    s = pid % S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * S + s
    x_ptrs = x_ptr + base + offs_c * S

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub
    sw = y * tl.sigmoid(y)
    sw = tl.where(mask_c, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)

    tl.store(out_ptr + n * S + s, out_val)


def fused_post(x, sub):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * S,)
    fused_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        num_warps=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.max_pool = nn.MaxPool3d(
            kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding,
        )
        self.subtract = nn.Parameter(torch.randn(out_channels))

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        self.pool_padding = pool_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()

        # Only handle the standard case via fused kernel
        if (self.pool_kernel_size == 2 and self.pool_stride == 2
                and self.pool_padding == 0):
            pooled = fused_conv_transpose_maxpool(
                x, w, b,
                self.stride, self.padding, self.output_padding,
                self.pool_kernel_size, self.pool_stride, self.pool_padding,
            )
        else:
            pooled = self.max_pool(self.conv_transpose(x))

        out = fused_post(pooled, self.subtract)
        return out