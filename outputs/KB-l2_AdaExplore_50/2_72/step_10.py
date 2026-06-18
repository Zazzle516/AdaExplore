import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _scatter_convt_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, ic, id, ih, iw)
    pid = tl.program_id(0)
    iw = pid % IW
    tmp = pid // IW
    ih = tmp % IH
    tmp = tmp // IH
    id_ = tmp % ID
    tmp = tmp // ID
    ic = tmp % IC
    n = tmp // IC

    x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
    x_val = tl.load(x_ptr + x_off)

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    for kd in range(0, KD):
        od = id_ * SD - PD + kd
        if (od >= 0) & (od < OD):
            for kh in range(0, KH):
                oh = ih * SH - PH + kh
                if (oh >= 0) & (oh < OH):
                    for kw in range(0, KW):
                        ow = iw * SW - PW + kw
                        if (ow >= 0) & (ow < OW):
                            # weight shape (IC, OC, KD, KH, KW)
                            w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                            w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                            contrib = x_val * w_val
                            out_off = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
                            tl.atomic_add(out_ptr + out_off, contrib, mask=oc_mask)


@triton.jit
def _bn_avgpool4_kernel(
    in_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C, ID, IH, IW,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    # one program per (n, c, od, oh tile of ow)
    pid = tl.program_id(0)
    total = N * C * OD * OH * OW
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow = offs % OW
    t1 = offs // OW
    oh = t1 % OH
    t2 = t1 // OH
    od = t2 % OD
    t3 = t2 // OD
    c = t3 % C
    n = t3 // C

    # avg pool kernel size 4, stride 4 (two avgpool2 stacked)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)

    for dd in range(0, 4):
        for hh in range(0, 4):
            for ww in range(0, 4):
                id_ = od * 4 + dd
                ih_ = oh * 4 + hh
                iw_ = ow * 4 + ww
                in_off = ((n * C + c) * ID + id_) * IH * IW + ih_ * IW + iw_
                v = tl.load(in_ptr + in_off, mask=mask, other=0.0)
                v = v * scale + shift
                acc = acc + v

    acc = acc / 64.0
    tl.store(out_ptr + offs, acc, mask=mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w
    SD = SH = SW = stride
    PD = PH = PW = padding
    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    # BLOCK_OC = next pow2 >= OC
    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = (N * IC * ID * IH * IW,)
    _scatter_convt_kernel[grid](
        x.contiguous(), weight.contiguous(), out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        num_warps=4,
    )

    if bias is not None:
        out = out + bias.view(1, OC, 1, 1, 1)
    return out


def fused_bn_avgpool(x, scale, shift):
    N, C, ID, IH, IW = x.shape
    OD, OH, OW = ID // 4, IH // 4, IW // 4
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=torch.float32)
    total = N * C * OD * OH * OW
    BLOCK = 128
    grid = ((total + BLOCK - 1) // BLOCK,)
    _bn_avgpool4_kernel[grid](
        x.contiguous(), out, scale.contiguous(), shift.contiguous(),
        N, C, ID, IH, IW, OD, OH, OW,
        BLOCK=BLOCK, num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight
        bias = self.conv_transpose.bias

        y = conv_transpose3d_triton(x, weight, bias, self.stride, self.padding)

        # batch norm fold
        if self.training:
            y = self.batch_norm(y)
            y = F.avg_pool3d(y, 2)
            y = F.avg_pool3d(y, 2)
            return y
        else:
            rm = self.batch_norm.running_mean
            rv = self.batch_norm.running_var
            eps = self.batch_norm.eps
            w = self.batch_norm.weight
            b = self.batch_norm.bias
            invstd = torch.rsqrt(rv + eps)
            scale = w * invstd
            shift = b - rm * scale
            return fused_bn_avgpool(y, scale, shift)