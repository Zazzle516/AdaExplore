import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, b_ptr, scale_ptr, bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    BLOCK_N: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    IC_C: tl.constexpr,
    KD_C: tl.constexpr,
    KH_C: tl.constexpr,
    KW_C: tl.constexpr,
):
    pid_n_block = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_batch = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    spatial_offs = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    spatial_mask = spatial_offs < ODHW

    od = spatial_offs // OHW
    rem = spatial_offs - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = b_vals[:, None] + tl.zeros([BLOCK_OC, BLOCK_N], dtype=tl.float32)

    IHW = IH * IW
    IDHW = ID * IHW
    KHW = KH * KW
    KDHW = KD * KHW

    x_batch_base = pid_batch * IC * IDHW
    w_oc_base = oc_offs * (IC_C * KDHW)  # [BLOCK_OC]

    for ic in tl.static_range(0, IC_C):
        x_ic_base = x_batch_base + ic * IDHW
        w_ic_base = w_oc_base + ic * KDHW
        for kd in tl.static_range(0, KD_C):
            id_ = od + kd
            x_d_base = x_ic_base + id_ * IHW
            w_d_base = w_ic_base + kd * KHW
            for kh in tl.static_range(0, KH_C):
                ih = oh + kh
                x_h_base = x_d_base + ih * IW
                w_h_base = w_d_base + kh * KW
                for kw in tl.static_range(0, KW_C):
                    iw = ow + kw
                    x_offs = x_h_base + iw
                    x_vals = tl.load(x_ptr + x_offs, mask=spatial_mask, other=0.0)
                    w_offs = w_h_base + kw
                    w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)
                    acc += w_vals[:, None] * x_vals[None, :]

    scale_vals = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    bias2_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    y = acc * scale_vals[:, None]
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    y = y * bias2_vals[:, None]
    y = tl.sigmoid(y)

    out_offs = pid_batch * OC * ODHW + oc_offs[:, None] * ODHW + spatial_offs[None, :]
    out_mask = oc_mask[:, None] & spatial_mask[None, :]
    tl.store(out_ptr + out_offs, y, mask=out_mask)


def conv3d_fused(x, weight, conv_bias, scale, bias):
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_N = 512
    BLOCK_OC = 16

    ODHW = OD * OH * OW
    grid = (
        (ODHW + BLOCK_N - 1) // BLOCK_N,
        (OC + BLOCK_OC - 1) // BLOCK_OC,
        N,
    )

    conv3d_fused_kernel[grid](
        x, weight, conv_bias,
        scale.contiguous().view(-1), bias.contiguous().view(-1),
        out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        BLOCK_N=BLOCK_N,
        BLOCK_OC=BLOCK_OC,
        IC_C=IC,
        KD_C=KD,
        KH_C=KH,
        KW_C=KW,
        num_warps=8,
        num_stages=3,
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
            x, self.conv.weight, self.conv.bias,
            self.scaling_factor, self.bias,
        )