import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [IC, OC, KD, KH, KW]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # one program per (n, od, oh, ow) -- each computes all OC outputs
    pid = tl.program_id(0)
    n = tl.program_id(1)

    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0).to(tl.float32)

    # for each kernel position, find input position
    # output[od, oh, ow] = sum_{ic, kd, kh, kw} input[ic, id_, ih_, iw_] * weight[ic, oc, kd, kh, kw]
    # where od = id_ * SD - PD + kd  =>  id_ = (od + PD - kd) / SD  if divisible
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = (id_num % SD == 0) & (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih = ih_num // SH
            ih_valid = (ih_num % SH == 0) & (ih >= 0) & (ih < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw = iw_num // SW
                iw_valid = (iw_num % SW == 0) & (iw >= 0) & (iw < IW)
                valid = id_valid & ih_valid & iw_valid
                if valid:
                    # sum over ic
                    for ic in range(0, IC):
                        x_off = ((n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw
                        xv = tl.load(x_ptr + x_off)
                        # weight[ic, oc, kd, kh, kw], shape [IC, OC, KD, KH, KW]
                        w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                        wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                        acc += xv * wv

    # store [N, OC, OD, OH, OW]
    out_base = ((n * OC + oc_offs) * OD + od) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_base, acc, mask=oc_mask)


@triton.jit
def fused_post_kernel(
    x_ptr,        # [N, C, S]  (S = D*H*W)
    out_ptr,      # [N, S]
    bias_ptr,     # scalar
    NS,           # N * S total spatial outputs
    C,
    S,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    s_start = pid * BLOCK_S
    s_offs = s_start + tl.arange(0, BLOCK_S)         # [BLOCK_S]
    s_mask = s_offs < NS                              # [BLOCK_S]

    n = s_offs // S
    s_in_n = s_offs % S

    c_offs = tl.arange(0, BLOCK_C)                    # [BLOCK_C]
    c_mask = c_offs < C

    # ptrs: [BLOCK_S, BLOCK_C]
    base = n * (C * S) + s_in_n                       # [BLOCK_S]
    ptrs = x_ptr + base[:, None] + c_offs[None, :] * S
    full_mask = s_mask[:, None] & c_mask[None, :]
    vals = tl.load(ptrs, mask=full_mask, other=-float('inf'))

    m = tl.max(vals, axis=1)                          # [BLOCK_S]
    e = tl.exp(vals - m[:, None])
    e = tl.where(full_mask, e, 0.0)
    s_sum = tl.sum(e, axis=1)                         # [BLOCK_S]
    lse = m + tl.log(s_sum)

    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    b = tl.load(bias_ptr)
    y = hs - b
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    tl.store(out_ptr + s_offs, y, mask=s_mask)


def conv_transpose3d_triton(x, weight, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    SD, SH, SW = stride, stride, stride
    PD, PH, PW = padding, padding, padding

    OD = (ID - 1) * SD - 2 * PD + KD
    OH = (IH - 1) * SH - 2 * PH + KH
    OW = (IW - 1) * SW - 2 * PW + KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=torch.float32)

    x_c = x.contiguous()
    w_c = weight.contiguous()
    b_c = bias.contiguous()

    BLOCK_OC = triton.next_power_of_2(OC)
    if BLOCK_OC < 16:
        BLOCK_OC = 16

    grid = (OD * OH * OW, N)
    conv_transpose3d_scatter_kernel[grid](
        x_c, w_c, b_c, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        SD=SD, SH=SH, SW=SW,
        PD=PD, PH=PH, PW=PW,
        BLOCK_OC=BLOCK_OC,
        num_warps=2,
    )
    return out


def fused_post(x, bias):
    N, C, D, H, W = x.shape
    x = x.contiguous()
    out = torch.empty((N, 1, D, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16
    BLOCK_S = 256
    S = D * H * W
    NS = N * S
    grid = (triton.cdiv(NS, BLOCK_S),)
    fused_post_kernel[grid](
        x, out, bias,
        NS, C, S,
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous()
        # Use torch's conv_transpose3d (cuDNN) — likely fastest for this shape
        y = F.conv_transpose3d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            stride=self.stride, padding=self.padding,
        )
        bias_scalar = self.bias.view(-1)[0:1]
        return fused_post(y, bias_scalar)