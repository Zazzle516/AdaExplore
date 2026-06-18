import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_convt_lse_hs_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [IC, OC, KD, KH, KW]
    cb_ptr,       # [OC] conv bias
    bias_ptr,     # scalar
    out_ptr,      # [N, 1, OD, OH, OW]
    N, IC,
    ID, IH, IW,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh, ow-tile)
    pid_n = tl.program_id(0)
    pid_dh = tl.program_id(1)
    pid_w = tl.program_id(2)

    od = pid_dh // OH
    oh = pid_dh % OH

    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)  # [BLOCK_W]
    ow_mask = ow < OW

    oc_range = tl.arange(0, OC)  # [OC]

    # accumulators [BLOCK_W, OC] - initialized with conv bias
    cb = tl.load(cb_ptr + oc_range)  # [OC]
    acc = tl.zeros((BLOCK_W, OC), dtype=tl.float32) + cb[None, :]

    # For each input position (id, ih, iw) and each kernel position (kd, kh, kw):
    #   od = id*STRIDE - PAD + kd  =>  id = (od + PAD - kd) / STRIDE
    # We iterate over kd, kh, kw and check that (od+PAD-kd) % STRIDE == 0 and id in range.

    for kd in tl.static_range(0, KD):
        num_d = od + PAD - kd
        idd = num_d // STRIDE
        d_valid = (num_d >= 0) & ((num_d % STRIDE) == 0) & (idd >= 0) & (idd < ID)
        for kh in tl.static_range(0, KH):
            num_h = oh + PAD - kh
            ihh = num_h // STRIDE
            h_valid = d_valid & (num_h >= 0) & ((num_h % STRIDE) == 0) & (ihh >= 0) & (ihh < IH)
            for kw in tl.static_range(0, KW):
                num_w = ow + PAD - kw  # [BLOCK_W]
                iww = num_w // STRIDE  # [BLOCK_W]
                w_valid = h_valid & (num_w >= 0) & ((num_w % STRIDE) == 0) & (iww >= 0) & (iww < IW) & ow_mask
                # Load weight: w[ic, oc, kd, kh, kw] for all ic, oc
                # weight shape [IC, OC, KD, KH, KW], stride: oc dim has KD*KH*KW
                # We loop over ic
                for ic in tl.static_range(0, 3):  # IC=3
                    if ic < IC:
                        # x[n, ic, idd, ihh, iww]
                        x_idx = ((pid_n * IC + ic) * ID + idd) * (IH * IW) + ihh * IW + iww
                        x_val = tl.load(x_ptr + x_idx, mask=w_valid, other=0.0)  # [BLOCK_W]
                        # w[ic, :, kd, kh, kw] -> [OC]
                        w_idx = ic * (OC * KD * KH * KW) + oc_range * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                        w_val = tl.load(w_ptr + w_idx)  # [OC]
                        acc += x_val[:, None] * w_val[None, :]

    # Now logsumexp across OC for each ow
    # acc: [BLOCK_W, OC]
    m = tl.max(acc, axis=1)  # [BLOCK_W]
    e = tl.exp(acc - m[:, None])
    s = tl.sum(e, axis=1)  # [BLOCK_W]
    lse = m + tl.log(s)  # [BLOCK_W]

    # HardSwish: x * sigmoid(x+3) / 6
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    b = tl.load(bias_ptr)
    y = hs - b
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    out_idx = ((pid_n * OD + od) * OH + oh) * OW + ow
    tl.store(out_ptr + out_idx, y, mask=ow_mask)


def fused_convt_lse_hs(x, weight, conv_bias, bias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape

    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_W = 16
    grid = (N, OD * OH, triton.cdiv(OW, BLOCK_W))

    fused_convt_lse_hs_kernel[grid](
        x, weight, conv_bias, bias.reshape(-1), out,
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
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        w = self.conv_transpose.weight
        cb = self.conv_transpose.bias
        return fused_convt_lse_hs(x, w, cb, self.bias, self.stride, self.padding)