import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_softmax_sigmoid_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow); compute all OC channels, then softmax+sigmoid
    pid = tl.program_id(0)
    spatial = OD * OH * OW
    n = pid // spatial
    s = pid % spatial
    od = s // (OH * OW)
    rem = s % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    offs_oc = tl.arange(0, BLOCK_OC)
    mask_oc = offs_oc < OC

    # initialize accumulator with bias
    acc = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0).to(tl.float32)

    # iterate over input channels and kernel positions
    # output[n, oc, od, oh, ow] = sum over ic, kd, kh, kw of:
    #    x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*SD = od + PD - kd, etc.

    for kd in range(KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = (id_num % SD == 0) & (id_ >= 0) & (id_ < ID)
        for kh in range(KH):
            ih_num = oh + PH - kh
            ih_ = ih_num // SH
            ih_valid = (ih_num % SH == 0) & (ih_ >= 0) & (ih_ < IH)
            for kw in range(KW):
                iw_num = ow + PW - kw
                iw_ = iw_num // SW
                iw_valid = (iw_num % SW == 0) & (iw_ >= 0) & (iw_ < IW)
                valid = id_valid & ih_valid & iw_valid
                if valid:
                    # accumulate over IC
                    for ic in range(IC):
                        x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih_ * IW + iw_
                        x_val = tl.load(x_ptr + x_off)
                        # weight shape: (IC, OC, KD, KH, KW)
                        w_off = ((ic * OC + offs_oc) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)
                        acc += x_val * w_val

    # softmax over OC
    acc = tl.where(mask_oc, acc, -float('inf'))
    m = tl.max(acc, axis=0)
    e = tl.exp(acc - m)
    e = tl.where(mask_oc, e, 0.0)
    s_sum = tl.sum(e, axis=0)
    sm = e / s_sum
    out = 1.0 / (1.0 + tl.exp(-sm))

    out_off = ((n * OC + offs_oc) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, out, mask=mask_oc)


def fused_conv_transpose_softmax_sigmoid(x, weight, bias, stride, padding, output_padding):
    assert x.is_cuda
    x = x.contiguous()
    weight = weight.contiguous()
    if bias is None:
        bias = torch.zeros(weight.shape[1], device=x.device, dtype=x.dtype)
    bias = bias.contiguous()

    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    SD, SH, SW = stride
    PD, PH, PW = padding
    OPD, OPH, OPW = output_padding

    OD = (ID - 1) * SD - 2 * PD + KD + OPD
    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2

    grid = (N * OD * OH * OW,)
    conv_transpose3d_softmax_sigmoid_kernel[grid](
        x, weight, bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding, bias=bias
        )
        ks = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        st = stride if isinstance(stride, tuple) else (stride,) * 3
        pd = padding if isinstance(padding, tuple) else (padding,) * 3
        op = output_padding if isinstance(output_padding, tuple) else (output_padding,) * 3
        self.ks = ks
        self.st = st
        self.pd = pd
        self.op = op

    def forward(self, x):
        return fused_conv_transpose_softmax_sigmoid(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            self.st, self.pd, self.op
        )