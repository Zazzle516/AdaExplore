import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, bias2_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # program_id(0) -> spatial tile index over OD*OH*OW
    # program_id(1) -> batch index
    # program_id(2) -> oc tile index
    pid_s = tl.program_id(0)
    n = tl.program_id(1)
    pid_oc = tl.program_id(2)

    s_offs = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    DHW = OD * OH * OW
    s_mask = s_offs < DHW
    oc_mask = oc_offs < OC

    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # init accumulator with bias
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = tl.zeros([BLOCK_OC, BLOCK_N], dtype=tl.float32) + b_vals[:, None]

    # input base for batch n
    x_batch = x_ptr + n * IC * ID * IH * IW

    for ic in tl.static_range(0, IC_C):
        ic_valid = ic < IC
        for kd in tl.static_range(0, KD):
            id_ = od + kd  # no padding
            for kh in tl.static_range(0, KH):
                ih = oh + kh
                for kw in tl.static_range(0, KW):
                    iw = ow + kw
                    # x: [BLOCK_N]
                    x_off = ic * (ID * IH * IW) + id_ * (IH * IW) + ih * IW + iw
                    x_vals = tl.load(x_batch + x_off, mask=s_mask & ic_valid, other=0.0)
                    # w: [BLOCK_OC]
                    w_off = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_off, mask=oc_mask & ic_valid, other=0.0)
                    acc += w_vals[:, None] * x_vals[None, :]

    # epilogue
    scale_vals = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    bias2_vals = tl.load(bias2_ptr + oc_offs, mask=oc_mask, other=0.0)

    y = acc * scale_vals[:, None]
    # tanh
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    y = y * bias2_vals[:, None]
    y = tl.sigmoid(y)

    # store: out[n, oc, od, oh, ow]
    out_off = n * (OC * DHW) + oc_offs[:, None] * DHW + s_offs[None, :]
    out_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=out_mask)


def conv3d_fused(x, weight, bias, scale, bias2):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_N = 128
    BLOCK_OC = 16  # OC=16, fits exactly
    DHW = OD * OH * OW
    grid = (triton.cdiv(DHW, BLOCK_N), N, triton.cdiv(OC, BLOCK_OC))

    # pad IC to power of 2 for static range (IC=3 -> use 4 with mask)
    IC_C = 4 if IC <= 4 else (8 if IC <= 8 else 16)

    conv3d_fused_kernel[grid](
        x, weight, bias, scale.view(-1), bias2.view(-1), out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        IC_C,
        BLOCK_N, BLOCK_OC,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return conv3d_fused(
            x,
            self.conv.weight,
            self.conv.bias,
            self.scaling_factor,
            self.bias,
        )