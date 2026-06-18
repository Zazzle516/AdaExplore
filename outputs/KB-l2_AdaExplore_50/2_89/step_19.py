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
    # one program per (n, pd, ph, pw); processes all OC channels in one go
    pid = tl.program_id(0)
    pw = pid % PW
    pid1 = pid // PW
    ph = pid1 % PH
    pid2 = pid1 // PH
    pd = pid2 % PD
    n = pid2 // PD

    od0 = pd * 2
    oh0 = ph * 2
    ow0 = pw * 2

    oc_offs = tl.arange(0, OC_C)
    oc_mask = oc_offs < OC

    # bias: [OC_C]
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)

    max_val = tl.full([OC_C], -float('inf'), dtype=tl.float32)

    ic_offs = tl.arange(0, IC_C)
    ic_mask = ic_offs < IC

    for ddi in tl.static_range(0, 2):
        for hhi in tl.static_range(0, 2):
            for wwi in tl.static_range(0, 2):
                od = od0 + ddi
                oh = oh0 + hhi
                ow = ow0 + wwi

                acc = bias  # [OC_C]

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

                            # x[n, ic, id, ih, iw]: [IC_C]
                            x_off = (((n * IC + ic_offs) * ID + id_val) * IH + ih_val) * IW + iw_val
                            xv = tl.load(x_ptr + x_off, mask=ic_mask & valid, other=0.0).to(tl.float32)

                            # weight[ic, oc, kd, kh, kw]: [IC_C, OC_C]
                            w_off = (((ic_offs[:, None] * OC + oc_offs[None, :]) * KD + kd) * KH + kh) * KW + kw
                            w_m = ic_mask[:, None] & oc_mask[None, :]
                            wv = tl.load(w_ptr + w_off, mask=w_m, other=0.0).to(tl.float32)

                            # contract IC: [IC_C, OC_C] * [IC_C, 1] -> sum over IC -> [OC_C]
                            acc += tl.sum(wv * xv[:, None], axis=0)

                max_val = tl.maximum(max_val, acc)

    # store to out[n, :, pd, ph, pw]
    out_off = (((n * OC + oc_offs) * PD + pd) * PH + ph) * PW + pw
    tl.store(out_ptr + out_off, max_val, mask=oc_mask)


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

    grid = (N * PD * PH * PW,)
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
        num_warps=4,
        num_stages=2,
    )
    return out


@triton.jit
def fused_softmax_sub_swish_max_kernel(
    x_ptr, sub_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < S

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # x: [BLOCK_C, BLOCK_S]
    x_ptrs = x_ptr + pid_n * C * S + offs_c[:, None] * S + offs_s[None, :]
    full_mask = mask_c[:, None] & mask_s[None, :]
    x = tl.load(x_ptrs, mask=full_mask, other=-float('inf'))

    m = tl.max(x, axis=0)  # [BLOCK_S]
    e = tl.exp(x - m[None, :])
    e = tl.where(full_mask, e, 0.0)
    s_sum = tl.sum(e, axis=0)  # [BLOCK_S]
    sm = e / s_sum[None, :]

    sub = tl.load(sub_ptr + offs_c, mask=mask_c, other=0.0)
    y = sm - sub[:, None]
    sw = y * tl.sigmoid(y)
    sw = tl.where(full_mask, sw, -float('inf'))
    out_val = tl.max(sw, axis=0)  # [BLOCK_S]

    out_ptrs = out_ptr + pid_n * S + offs_s
    tl.store(out_ptrs, out_val, mask=mask_s)


def fused_post(x, sub):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = 128
    grid = (N, triton.cdiv(S, BLOCK_S))
    fused_softmax_sub_swish_max_kernel[grid](
        x_c, sub.contiguous(), out,
        N, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
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