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
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    kD: tl.constexpr, kH: tl.constexpr, kW: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
):
    # one program per (n, oc, pd, ph, pw)
    pid = tl.program_id(0)
    pw = pid % PW
    tmp = pid // PW
    ph = tmp % PH
    tmp = tmp // PH
    pd = tmp % PD
    tmp = tmp // PD
    oc = tmp % OC_C
    n = tmp // OC_C

    # Pool window: output coords [pd*2 .. pd*2+1] etc
    od0 = pd * 2
    oh0 = ph * 2
    ow0 = pw * 2

    max_val = -float('inf')

    # iterate over 8 positions in pool window
    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            for ww in tl.static_range(0, 2):
                od = od0 + dd
                oh = oh0 + hh
                ow = ow0 + ww

                acc = 0.0
                # iterate over kernel
                for kd in tl.static_range(0, kD):
                    id_nom = od + pad_d - kd
                    id_ = id_nom // stride_d
                    id_valid = (id_nom % stride_d == 0) & (id_ >= 0) & (id_ < ID)
                    for kh in tl.static_range(0, kH):
                        ih_nom = oh + pad_h - kh
                        ih = ih_nom // stride_h
                        ih_valid = (ih_nom % stride_h == 0) & (ih >= 0) & (ih < IH)
                        for kw in tl.static_range(0, kW):
                            iw_nom = ow + pad_w - kw
                            iw = iw_nom // stride_w
                            iw_valid = (iw_nom % stride_w == 0) & (iw >= 0) & (iw < IW)
                            valid = id_valid & ih_valid & iw_valid

                            # accumulate over IC
                            offs_ic = tl.arange(0, IC_C)
                            mask_ic = offs_ic < IC

                            # x[n, ic, id, ih, iw]
                            x_off = ((n * IC + offs_ic) * ID + id_) * IH * IW + ih * IW + iw
                            x_vals = tl.load(x_ptr + x_off, mask=mask_ic & valid, other=0.0)

                            # w[ic, oc, kd, kh, kw]
                            w_off = ((offs_ic * OC + oc) * kD + kd) * kH * kW + kh * kW + kw
                            w_vals = tl.load(w_ptr + w_off, mask=mask_ic, other=0.0)

                            acc += tl.sum(x_vals * w_vals, axis=0)

                bias = tl.load(b_ptr + oc)
                acc = acc + bias
                max_val = tl.maximum(max_val, acc)

    out_off = ((n * OC + oc) * PD + pd) * PH * PW + ph * PW + pw
    tl.store(out_ptr + out_off, max_val)


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


def fused_convtranspose_maxpool(x, weight, bias, stride, padding, output_padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, kD, kH, kW = weight.shape

    # Output dims of conv_transpose
    OD = (ID - 1) * stride[0] - 2 * padding[0] + kD + output_padding[0]
    OH = (IH - 1) * stride[1] - 2 * padding[1] + kH + output_padding[1]
    OW = (IW - 1) * stride[2] - 2 * padding[2] + kW + output_padding[2]

    # Pool dims (kernel=2, stride=2, padding=0)
    PD = OD // 2
    PH = OH // 2
    PW = OW // 2

    out = torch.empty((N, OC, PD, PH, PW), device=x.device, dtype=x.dtype)

    IC_C = triton.next_power_of_2(IC)

    grid = (N * OC * PD * PH * PW,)
    conv_transpose_pool_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        PD, PH, PW,
        stride[0], stride[1], stride[2],
        padding[0], padding[1], padding[2],
        kD=kD, kH=kH, kW=kW,
        IC_C=IC_C,
        OC_C=OC,
        num_warps=1,
    )
    return out


def fused_post(x, sub):
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * S,)
    fused_softmax_sub_swish_max_kernel[grid](
        x, sub.contiguous(), out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        num_warps=1,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, pool_stride, pool_padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.max_pool = nn.MaxPool3d(kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding)
        self.subtract = nn.Parameter(torch.randn(out_channels))

        self.stride_t = (stride, stride, stride) if isinstance(stride, int) else stride
        self.padding_t = (padding, padding, padding) if isinstance(padding, int) else padding
        self.output_padding_t = (output_padding, output_padding, output_padding) if isinstance(output_padding, int) else output_padding
        self.kernel_size_t = (kernel_size, kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()
        x = fused_convtranspose_maxpool(x, w, b, self.stride_t, self.padding_t, self.output_padding_t)
        x = fused_post(x, self.subtract)
        return x