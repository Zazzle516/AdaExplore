import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, add_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    OHW = OH * OW
    ODHW = OD * OHW

    sp_mask = sp_offs < ODHW
    oc_mask = oc_offs < OC

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # We need to accumulate over ic, kd, kh, kw where:
    # od + PD = id*SD + kd  =>  kd = od + PD - id*SD,  id = (od + PD - kd) / SD
    # so for each (kd), id = (od+PD-kd)/SD must be integer in [0, ID), kd in [0,KD)
    # Iterate over kd, kh, kw, ic.

    acc = tl.zeros([BLOCK_SP, BLOCK_OC], dtype=tl.float32)

    od_p = od + PD  # [BLOCK_SP]
    oh_p = oh + PH
    ow_p = ow + PW

    for kd in tl.static_range(0, KD):
        id_num = od_p - kd
        id_ = id_num // SD
        id_valid = (id_num - id_ * SD == 0) & (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh_p - kh
            ih_ = ih_num // SH
            ih_valid = (ih_num - ih_ * SH == 0) & (ih_ >= 0) & (ih_ < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow_p - kw
                iw_ = iw_num // SW
                iw_valid = (iw_num - iw_ * SW == 0) & (iw_ >= 0) & (iw_ < IW)
                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask

                # input offset (without ic): n*IC*ID*IH*IW + 0*... + id*IH*IW + ih*IW + iw
                in_spatial_off = id_ * (IH * IW) + ih_ * IW + iw_  # [BLOCK_SP]
                # weight offset (without ic, oc): ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                w_kernel_off = kd * (KH * KW) + kh * KW + kw

                # Loop over ic
                for ic in range(0, IC):
                    x_off = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + in_spatial_off
                    x_vals = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_kernel_off
                    w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_vals[:, None] * w_vals[None, :]

    # Add the add_input and apply x * hardswish(x)
    # output offset: n*OC*ODHW + oc*ODHW + sp
    out_off = pid_n * (OC * ODHW) + oc_offs[None, :] * ODHW + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]

    add_vals = tl.load(add_ptr + out_off, mask=out_mask, other=0.0)
    v = acc + add_vals
    hs = v * tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0) * (1.0 / 6.0)
    res = v * hs
    tl.store(out_ptr + out_off, res, mask=out_mask)


def conv_transpose3d_fused(x, weight, add_input, stride, padding, output_padding):
    N, IC, ID, IH, IW = x.shape
    _, OC, KD, KH, KW = weight.shape
    SD, SH, SW = stride, stride, stride
    PD, PH, PW = padding, padding, padding
    OPD, OPH, OPW = output_padding, output_padding, output_padding

    OD = (ID - 1) * SD - 2 * PD + KD + OPD
    OH = (IH - 1) * SH - 2 * PH + KH + OPH
    OW = (IW - 1) * SW - 2 * PW + KW + OPW

    x = x.contiguous()
    weight = weight.contiguous()
    add_input = add_input.contiguous()
    out = torch.empty((N, OC, OD, OH, OW), dtype=x.dtype, device=x.device)

    BLOCK_OC = 32
    BLOCK_SP = 64

    grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OD * OH * OW, BLOCK_SP))

    conv_transpose3d_fused_kernel[grid](
        x, weight, add_input, out,
        N, IC, OC,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        SD, SH, SW,
        PD, PH, PW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_SP=BLOCK_SP,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x, add_input):
        # weight shape: (in_channels, out_channels, kD, kH, kW)
        # We need to also add the conv bias if present
        weight = self.conv_transpose.weight
        conv_bias = self.conv_transpose.bias

        if conv_bias is not None:
            # Fold conv bias into add_input (broadcast along channel)
            add_input_eff = add_input + conv_bias.view(1, -1, 1, 1, 1)
        else:
            add_input_eff = add_input

        out = conv_transpose3d_fused(
            x, weight, add_input_eff,
            self.stride, self.padding, self.output_padding
        )
        return out