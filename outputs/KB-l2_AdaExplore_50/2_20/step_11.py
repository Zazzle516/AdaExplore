import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_conv_ptr, b_extra_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    SP_TOTAL = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)

    sp_mask = sp_offs < SP_TOTAL
    oc_mask = oc_offs < OC

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # accumulator [BLOCK_SP, BLOCK_OC]
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each kernel position
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_q = id_num // SD
        id_valid = ((id_num - id_q * SD) == 0) & (id_q >= 0) & (id_q < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_valid = ((ih_num - ih_q * SH) == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_q = iw_num // SW
                iw_valid = ((iw_num - iw_q * SW) == 0) & (iw_q >= 0) & (iw_q < IW)
                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # input offset: n * IC * ID * IH * IW + ic * ID*IH*IW + id_q*IH*IW + ih_q*IW + iw_q
                in_spatial = id_q * (IH * IW) + ih_q * IW + iw_q  # [BLOCK_SP]

                # Loop over input channels
                for ic in range(0, IC):
                    in_off = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + in_spatial
                    x_val = tl.load(x_ptr + in_off, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    # weight offset: ic * OC * KD*KH*KW + oc * KD*KH*KW + kd*KH*KW + kh*KW + kw
                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_val[:, None] * w_val[None, :]

    # Add conv bias
    bc = tl.load(b_conv_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + bc[None, :]

    # Now acc = conv result. Apply epilogue: (2x + b_extra)*x + x
    be = tl.load(b_extra_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    x_full = acc
    y = (2.0 * x_full + be[None, :]) * x_full + x_full

    # store: out[n, oc, od, oh, ow]
    out_off = (pid_n * OC * SP_TOTAL
               + oc_offs[None, :] * SP_TOTAL
               + sp_offs[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OPD = OPH = OPW = self.output_padding

        OD = (ID - 1) * SD - 2 * PD + KD + OPD
        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW
        OC = self.out_channels

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        b_conv = self.conv_transpose.bias.contiguous()
        b_extra = self.bias.contiguous().view(-1)

        BLOCK_OC = 64
        BLOCK_SP = 64
        SP_TOTAL = OD * OH * OW

        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (SP_TOTAL + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose3d_kernel[grid](
            x, weight, b_conv, b_extra, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out