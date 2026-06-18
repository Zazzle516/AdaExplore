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
):
    # Each program computes BLOCK_N output spatial positions for BLOCK_OC channels (one batch)
    pid_n_block = tl.program_id(0)  # spatial block id
    pid_oc = tl.program_id(1)       # oc tile id
    pid_batch = tl.program_id(2)    # batch id

    OHW = OH * OW
    ODHW = OD * OHW

    spatial_offs = pid_n_block * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    spatial_mask = spatial_offs < ODHW

    od = spatial_offs // OHW
    rem = spatial_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # Load bias once
    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    # acc init with bias broadcast: [BLOCK_OC, BLOCK_N]
    acc = b_vals[:, None] + tl.zeros([BLOCK_OC, BLOCK_N], dtype=tl.float32)

    IHW = IH * IW
    IDHW = ID * IHW

    x_batch_base = pid_batch * IC * IDHW

    # Loop over IC, KD, KH, KW
    for ic in range(0, IC):
        for kd in range(0, KD):
            id_ = od + kd  # [BLOCK_N]
            for kh in range(0, KH):
                ih = oh + kh  # [BLOCK_N]
                for kw in range(0, KW):
                    iw = ow + kw  # [BLOCK_N]

                    x_offs = x_batch_base + ic * IDHW + id_ * IHW + ih * IW + iw
                    x_vals = tl.load(x_ptr + x_offs, mask=spatial_mask, other=0.0)  # [BLOCK_N]

                    # weight offset: oc * (IC*KD*KH*KW) + ic*(KD*KH*KW) + kd*(KH*KW) + kh*KW + kw
                    w_offs = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_vals[:, None] * x_vals[None, :]

    # Epilogue: scale, tanh, bias, sigmoid
    scale_vals = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    bias2_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)   # [BLOCK_OC]

    y = acc * scale_vals[:, None]
    # tanh via sigmoid
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    y = y * bias2_vals[:, None]
    y = tl.sigmoid(y)

    # store: out[batch, oc, spatial]
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

    BLOCK_N = 128
    BLOCK_OC = 16  # OC=16 exactly

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
        num_warps=4,
        num_stages=2,
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