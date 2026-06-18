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
    # program ids: (n, od*oh*ow tile? we do n, oc_tile, spatial)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_spatial = tl.program_id(2)

    od = pid_spatial // (OH * OW)
    rem = pid_spatial % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # For ConvTranspose3d: y[n,oc,od,oh,ow] = sum_{ic,kd,kh,kw} x[n,ic,id,ih,iw] * w[ic,oc,kd,kh,kw]
    # where: od = id*SD - PD + kd  =>  id = (od + PD - kd) / SD, must be integer and in range
    for kd in range(KD):
        id_num = od + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_val >= 0) & (id_val < ID)
        for kh in range(KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in range(KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_val >= 0) & (iw_val < IW)

                spatial_valid = id_valid & ih_valid & iw_valid

                for ic in range(IC):
                    # load x[n, ic, id_val, ih_val, iw_val] - scalar
                    x_off = ((pid_n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                    x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)
                    # load w[ic, oc_offs, kd, kh, kw]
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += x_val * w_val

    # add bias
    bias_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias_val

    # store
    out_off = ((pid_n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


@triton.jit
def fused_scale_pool_bias_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    scale1, scale2,
    N, C, ID, IH, IW,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    # one program per output element batch
    pid = tl.program_id(0)
    total = N * C * OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # decode indices
    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    t2 = t1 // OH
    od = t2 % OD
    t3 = t2 // OD
    c = t3 % C
    n = t3 // C

    # input indices: 2x window starting at (2*od, 2*oh, 2*ow)
    id0 = od * 2
    ih0 = oh * 2
    iw0 = ow * 2

    base = ((n * C + c) * ID) * IH * IW

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for dd in range(2):
        for hh in range(2):
            for ww in range(2):
                idx = base + (id0 + dd) * IH * IW + (ih0 + hh) * IW + (iw0 + ww)
                v = tl.load(x_ptr + idx, mask=mask, other=0.0)
                acc += v

    acc = acc * (scale1 / 8.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    acc = (acc + b) * scale2

    tl.store(out_ptr + offs, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    SD = SH = SW = stride if isinstance(stride, int) else stride[0]
    PD = PH = PW = padding if isinstance(padding, int) else padding[0]

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 16
    grid = (N, triton.cdiv(OC, BLOCK_OC), OD * OH * OW)

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


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale1, scale2, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale1 = nn.Parameter(torch.tensor(scale1))
        self.avg_pool = nn.AvgPool3d(kernel_size=2)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scale2 = nn.Parameter(torch.tensor(scale2))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous()
        b = self.conv_transpose.bias.contiguous()

        # Custom conv transpose 3d
        y = conv_transpose3d_triton(x, w, b, self.stride, self.padding)

        # Fused: scale1, avgpool(2), bias add, scale2
        N, C, ID, IH, IW = y.shape
        OD, OH, OW = ID // 2, IH // 2, IW // 2

        out = torch.empty((N, C, OD, OH, OW), device=y.device, dtype=y.dtype)

        total = N * C * OD * OH * OW
        BLOCK = 256
        grid = (triton.cdiv(total, BLOCK),)

        bias_flat = self.bias.contiguous().view(-1)

        fused_scale_pool_bias_scale_kernel[grid](
            y, bias_flat, out,
            float(self.scale1.item()), float(self.scale2.item()),
            N, C, ID, IH, IW,
            OD, OH, OW,
            BLOCK=BLOCK,
        )
        return out