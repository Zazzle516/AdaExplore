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
    # program ids: (n, od, oh*ow tile, oc tile)
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_spatial = tl.program_id(2)
    pid_oc = tl.program_id(3)

    oh = pid_spatial // OW
    ow = pid_spatial % OW
    od = pid_d

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # For ConvTranspose3d:
    # output[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw) of
    #   x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*SD - PD + kd = od => id = (od + PD - kd) / SD
    # and (od + PD - kd) must be divisible by SD, and id must be in [0, ID)

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    for kd in range(0, KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num % SD) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in range(0, KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num % SH) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in range(0, KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num % SW) == 0) & (iw_val >= 0) & (iw_val < IW)
                valid = id_valid & ih_valid & iw_valid
                for ic in range(0, IC):
                    # x[n, ic, id_val, ih_val, iw_val]
                    x_off = ((pid_n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    # w[ic, oc, kd, kh, kw]
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += x_val * w_val

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias

    out_off = ((pid_n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


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
    grid = (N, OD, OH * OW, triton.cdiv(OC, BLOCK_OC))
    conv_transpose3d_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
    )
    return out


@triton.jit
def fused_pool_bias_scale_kernel(
    in_ptr, bias_ptr, out_ptr,
    scale_combined,
    N, C, D, H, W,
    PD, PH, PW,  # pooled dims
    BLOCK: tl.constexpr,
):
    # each program handles BLOCK output elements
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * PD * PH * PW
    mask = offs < total

    # decode
    pw = offs % PW
    tmp = offs // PW
    ph = tmp % PH
    tmp = tmp // PH
    pd = tmp % PD
    tmp = tmp // PD
    c = tmp % C
    n = tmp // C

    # 2x2x2 pool: input region [pd*2:pd*2+2, ph*2:ph*2+2, pw*2:pw*2+2]
    d0 = pd * 2
    h0 = ph * 2
    w0 = pw * 2

    base = ((n * C + c) * D + d0) * H * W + h0 * W + w0

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in range(0, 2):
        for hh in range(0, 2):
            for ww in range(0, 2):
                off = base + dd * H * W + hh * W + ww
                v = tl.load(in_ptr + off, mask=mask, other=0.0)
                acc += v

    acc = acc * (scale_combined / 8.0)  # avg = sum/8, then *scale1*scale2
    # but scale1 already applied? We pass combined = scale1 * scale2
    # Actually: after conv, x*scale1, pool (avg), +bias, *scale2
    # = ((sum/8)*scale1 + bias) * scale2
    # = sum * scale1*scale2/8 + bias*scale2
    # So we need bias*scale2 too

    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    # acc currently = sum * scale_combined/8 where scale_combined = scale1*scale2
    acc = acc + b  # b is already bias*scale2 (passed in pre-multiplied)

    tl.store(out_ptr + offs, acc, mask=mask)


def fused_pool_bias_scale(x, bias_scaled, scale_combined):
    N, C, D, H, W = x.shape
    PD, PH, PW = D // 2, H // 2, W // 2
    out = torch.empty((N, C, PD, PH, PW), device=x.device, dtype=x.dtype)
    total = N * C * PD * PH * PW
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    fused_pool_bias_scale_kernel[grid](
        x, bias_scaled, out,
        scale_combined,
        N, C, D, H, W,
        PD, PH, PW,
        BLOCK=BLOCK,
        num_warps=4,
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
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv_transpose.weight.contiguous()
        conv_bias = self.conv_transpose.bias.contiguous()

        # Do conv transpose with bias added
        y = conv_transpose3d_triton(x, weight, conv_bias, self.stride, self.padding)

        # Fused: scale1 * avg_pool + bias, * scale2
        # = (sum/8 * scale1 + bias) * scale2
        # = sum * (scale1*scale2/8) + bias*scale2
        scale_combined = (self.scale1 * self.scale2).item()
        bias_scaled = (self.bias * self.scale2).reshape(-1).contiguous()

        out = fused_pool_bias_scale(y, bias_scaled, scale_combined)
        return out