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
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_DHW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_dhw = tl.program_id(2)

    dhw_offs = pid_dhw * BLOCK_DHW + tl.arange(0, BLOCK_DHW)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    OHW = OH * OW
    ODHW = OD * OHW

    dhw_mask = dhw_offs < ODHW
    oc_mask = oc_offs < OC

    od = dhw_offs // OHW
    rem = dhw_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # load bias for conv (per OC)
    conv_b = tl.load(w_ptr + 0, mask=False, other=0.0)  # placeholder unused

    # Accumulator [BLOCK_OC, BLOCK_DHW]
    acc = tl.zeros((BLOCK_OC, BLOCK_DHW), dtype=tl.float32)

    # Loop over IC, KD, KH, KW
    IDHW = ID * IH * IW
    IHW = IH * IW
    x_base = pid_n * IC * IDHW

    for ic in range(0, IC):
        for kd in tl.static_range(0, KD):
            id_ = od + kd  # padding=0
            for kh in tl.static_range(0, KH):
                ih_ = oh + kh
                for kw in tl.static_range(0, KW):
                    iw_ = ow + kw
                    # input index per dhw
                    in_idx = x_base + ic * IDHW + id_ * IHW + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + in_idx, mask=dhw_mask, other=0.0)  # [BLOCK_DHW]

                    # weight index per oc: w[oc, ic, kd, kh, kw]
                    w_idx = oc_offs * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    # Apply scaling -> tanh -> bias -> sigmoid
    scale_val = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)
    bias_val = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)

    y = acc * scale_val[:, None]
    # tanh = 2*sigmoid(2x) - 1
    y = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    y = y * bias_val[:, None]
    y = tl.sigmoid(y)

    # Store: out[n, oc, od, oh, ow]
    out_base = pid_n * OC * ODHW
    out_idx = out_base + oc_offs[:, None] * ODHW + dhw_offs[None, :]
    out_mask = oc_mask[:, None] & dhw_mask[None, :]
    tl.store(out_ptr + out_idx, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_DHW = 128
        BLOCK_OC = 16  # OC=16

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()
        scale = self.scaling_factor.contiguous().view(-1)
        bias_param = self.bias.contiguous().view(-1)

        ODHW = OD * OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (ODHW + BLOCK_DHW - 1) // BLOCK_DHW)

        conv3d_fused_kernel[grid](
            x, weight, bias, scale, bias_param, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_DHW=BLOCK_DHW,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )
        return out