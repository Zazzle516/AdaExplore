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
    IC_C: tl.constexpr, OC_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # program ids
    pid_m = tl.program_id(0)  # tile of output spatial positions
    pid_n = tl.program_id(1)  # batch index

    # output spatial linear index range
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < (OD * OH * OW)

    # decompose linear m into (od, oh, ow)
    od = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    n = pid_n

    # accumulator: [BLOCK_M, OC_C]
    acc = tl.zeros((BLOCK_M, OC_C), dtype=tl.float32)

    # weight shape: [OC, IC, KD, KH, KW]
    # Loop over IC * KD * KH * KW
    oc_offs = tl.arange(0, OC_C)  # [OC_C]

    for ic in tl.static_range(0, IC_C):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    # input position
                    id_pos = od + kd
                    ih_pos = oh + kh
                    iw_pos = ow + kw

                    # input offset: ((n*IC + ic)*ID + id_pos)*IH*IW + ih_pos*IW + iw_pos
                    in_off = ((n * IC_C + ic) * ID + id_pos) * (IH * IW) + ih_pos * IW + iw_pos
                    x_vals = tl.load(x_ptr + in_off, mask=m_mask, other=0.0)  # [BLOCK_M]

                    # weight offset: ((oc*IC + ic)*KD + kd)*KH*KW + kh*KW + kw
                    w_off = ((oc_offs * IC_C + ic) * KD + kd) * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_off)  # [OC_C]

                    acc += x_vals[:, None] * w_vals[None, :]

    # bias
    b_vals = tl.load(b_ptr + oc_offs)  # [OC_C]
    acc = acc + b_vals[None, :]

    # scaling_factor (per OC)
    s_vals = tl.load(scale_ptr + oc_offs)  # [OC_C]
    y = acc * s_vals[None, :]

    # tanh
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)

    # bias2
    b2_vals = tl.load(bias2_ptr + oc_offs)  # [OC_C]
    z = t * b2_vals[None, :]

    # sigmoid
    out = 1.0 / (1.0 + tl.exp(-z))

    # store: output layout [N, OC, OD, OH, OW]
    # out_off = ((n*OC + oc)*OD + od)*OH*OW + oh*OW + ow
    out_off = ((n * OC_C + oc_offs[None, :]) * OD + od[:, None]) * (OH * OW) + oh[:, None] * OW + ow[:, None]
    out_mask = m_mask[:, None] & (oc_offs[None, :] < OC_C)
    tl.store(out_ptr + out_off, out, mask=out_mask)


def fused_conv3d(x, weight, bias, scale, bias2):
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    scale = scale.contiguous().view(-1)
    bias2 = bias2.contiguous().view(-1)

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    OC_C = OC  # 16
    IC_C = IC  # 3

    grid = (triton.cdiv(OD * OH * OW, BLOCK_M), N)

    conv3d_fused_kernel[grid](
        x, weight, bias, scale, bias2, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        IC_C, OC_C,
        BLOCK_M=BLOCK_M,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        return fused_conv3d(
            x, self.conv.weight, self.conv.bias,
            self.scaling_factor, self.bias,
        )