import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow) tile over OC
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)

    od = pid_spatial // (OH * OW)
    rem = pid_spatial % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # for each input contribution: x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where: od + PD = id * SD + kd
    # so kd = od + PD - id * SD, with 0 <= kd < KD
    # id range: id_min = ceil((od + PD - (KD-1)) / SD), id_max = (od + PD) / SD
    d_top = od + PD
    h_top = oh + PH
    w_top = ow + PW

    # iterate over kd, kh, kw
    for kd in range(KD):
        id_num = d_top - kd
        # need id_num >= 0 and id_num % SD == 0 and id_num // SD < ID
        id_val = id_num // SD
        valid_d = (id_num >= 0) & ((id_num % SD) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in range(KH):
            ih_num = h_top - kh
            ih_val = ih_num // SH
            valid_h = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in range(KW):
                iw_num = w_top - kw
                iw_val = iw_num // SW
                valid_w = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val >= 0) & (iw_val < IW)
                valid = valid_d & valid_h & valid_w

                if valid:
                    # sum over IC: x[n, ic, id_val, ih_val, iw_val] * w[ic, oc, kd, kh, kw]
                    for ic in range(IC):
                        x_idx = ((n_index := pid_n) * IC + ic) * ID * IH * IW + id_val * IH * IW + ih_val * IW + iw_val
                        x_val = tl.load(x_ptr + x_idx)
                        w_idx = (ic * OC + oc_offs) * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                        acc += x_val * w_vals

    # add bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals

    out_idx = ((pid_n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_idx, acc, mask=oc_mask)


@triton.jit
def fused_scale_avgpool_bias_scale_kernel(
    in_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    scale1, scale2,
    BLOCK: tl.constexpr,
):
    # one program per output element block
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OD * OH * OW
    mask = offs < total

    # decode index
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    # input starting indices
    id0 = od * 2
    ih0 = oh * 2
    iw0 = ow * 2

    base = ((n * C + c) * D + id0) * H * W + ih0 * W + iw0
    stride_h = W
    stride_d = H * W

    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in range(2):
        for hh in range(2):
            for ww in range(2):
                idx = base + dd * stride_d + hh * stride_h + ww
                v = tl.load(in_ptr + idx, mask=mask, other=0.0)
                s += v

    avg = s * (scale1 / 8.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    res = (avg + b) * scale2
    tl.store(out_ptr + offs, res, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    grid = (N, OD * OH * OW, (OC + BLOCK_OC - 1) // BLOCK_OC)

    # We need to inline pid_n - the kernel uses walrus, but let's just pass differently
    # Actually walrus may not work in triton. Let's not use it.
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
    )
    return out


# Rewrite kernel without walrus operator (which may not be supported)
@triton.jit
def conv_transpose3d_kernel_v2(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_spatial = tl.program_id(1)
    pid_oc = tl.program_id(2)

    od = pid_spatial // (OH * OW)
    rem = pid_spatial % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    d_top = od + PD
    h_top = oh + PH
    w_top = ow + PW

    for kd in range(KD):
        id_num = d_top - kd
        id_val = id_num // SD
        valid_d = (id_num >= 0) & ((id_num % SD) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in range(KH):
            ih_num = h_top - kh
            ih_val = ih_num // SH
            valid_h = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in range(KW):
                iw_num = w_top - kw
                iw_val = iw_num // SW
                valid_w = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val >= 0) & (iw_val < IW)
                valid = valid_d & valid_h & valid_w

                if valid:
                    for ic in range(IC):
                        x_idx = ((pid_n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                        x_val = tl.load(x_ptr + x_idx)
                        w_idx = (ic * OC + oc_offs) * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        w_vals = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                        acc += x_val * w_vals

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals

    out_idx = ((pid_n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_idx, acc, mask=oc_mask)


def conv_transpose3d_triton_v2(x, weight, bias, stride, padding):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride
    PD = PH = PW = padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    grid = (N, OD * OH * OW, (OC + BLOCK_OC - 1) // BLOCK_OC)

    conv_transpose3d_kernel_v2[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
    )
    return out


def fused_post_triton(x, bias, scale1, scale2):
    x = x.contiguous()
    bias = bias.contiguous().view(-1)
    N, C, D, H, W = x.shape
    OD, OH, OW = D // 2, H // 2, W // 2
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OD * OH * OW
    BLOCK = 256
    grid = ((total + BLOCK - 1) // BLOCK,)
    fused_scale_avgpool_bias_scale_kernel[grid](
        x, bias, out,
        N, C, D, H, W,
        OD, OH, OW,
        float(scale1), float(scale2),
        BLOCK=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias
        y = conv_transpose3d_triton_v2(x, weight, conv_bias, self.stride, self.padding)
        out = fused_post_triton(y, self.bias, self.scale1.item(), self.scale2.item())
        return out