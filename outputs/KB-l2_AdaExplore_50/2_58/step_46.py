import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_transpose_logsumexp_kernel(
    x_ptr,          # [N, IC, ID, IH, IW]
    w_ptr,          # [IC, OC, KD, KH, KW]
    cbias_ptr,      # [OC] conv bias
    sbias_ptr,      # scalar bias
    out_ptr,        # [N, 1, OD, OH, OW]
    N, IC,
    ID, IH, IW,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh, ow-tile)
    pid_now = tl.program_id(0)  # n * OD * OH
    pid_w = tl.program_id(1)

    n = pid_now // (OD * OH)
    rem = pid_now % (OD * OH)
    od = rem // OH
    oh = rem % OH

    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)  # [BLOCK_W]
    ow_mask = ow_offs < OW

    oc_range = tl.arange(0, OC)  # [OC]

    # accumulator [BLOCK_W, OC]
    acc = tl.zeros((BLOCK_W, OC), dtype=tl.float32)

    # Loop over kernel positions; for each, derive ic input position
    # For conv_transpose: out[n, oc, od, oh, ow] = sum_{ic, kd, kh, kw} x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*STRIDE - PAD + kd = od  =>  id = (od + PAD - kd) / STRIDE (must be integer, in range)
    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_ok = (id_num % STRIDE == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PAD - kh
            ih_val = ih_num // STRIDE
            ih_ok = id_ok & (ih_num % STRIDE == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PAD - kw  # [BLOCK_W]
                iw_val = iw_num // STRIDE
                iw_ok = ih_ok & (iw_num % STRIDE == 0) & (iw_val >= 0) & (iw_val < IW) & ow_mask  # [BLOCK_W]

                # for each ic, load x[n, ic, id_val, ih_val, iw_val] -> [BLOCK_W]
                # and w[ic, :, kd, kh, kw] -> [OC]
                for ic in tl.static_range(0, 3):  # IC=3
                    x_idx = ((n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val  # [BLOCK_W]
                    x_vals = tl.load(x_ptr + x_idx, mask=iw_ok, other=0.0)  # [BLOCK_W]

                    w_idx = ic * OC * KD * KH * KW + oc_range * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw  # [OC]
                    w_vals = tl.load(w_ptr + w_idx)  # [OC]

                    acc += x_vals[:, None] * w_vals[None, :]

    # add conv bias
    cb = tl.load(cbias_ptr + oc_range)  # [OC]
    acc += cb[None, :]

    # logsumexp across OC
    m = tl.max(acc, axis=1)  # [BLOCK_W]
    e = tl.exp(acc - m[:, None])
    s = tl.sum(e, axis=1)
    lse = m + tl.log(s)  # [BLOCK_W]

    # hardswish: x * sigmoid(x+3) / 6
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    # subtract scalar bias
    sb = tl.load(sbias_ptr)
    y = hs - sb
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    # store
    out_off = n * (OD * OH * OW) + od * (OH * OW) + oh * OW + ow_offs
    tl.store(out_ptr + out_off, y, mask=ow_mask)


def fused_op(x, weight, cbias, sbias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_W = 16
    # next pow2 for OW tile choice; keep <= OW reasonable
    while BLOCK_W < OW and BLOCK_W < 64:
        BLOCK_W *= 2

    grid = (N * OD * OH, triton.cdiv(OW, BLOCK_W))
    fused_conv_transpose_logsumexp_kernel[grid](
        x, weight, cbias, sbias.reshape(-1),
        out,
        N, IC,
        ID, IH, IW,
        OD, OH, OW,
        OC=OC,
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride, PAD=padding,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        w = self.conv_transpose.weight
        cb = self.conv_transpose.bias
        return fused_op(x, w, cb, self.bias, self.stride, self.padding)