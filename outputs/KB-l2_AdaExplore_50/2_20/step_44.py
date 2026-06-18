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
    SP_total = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < SP_total

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # accumulator
    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # For each output (od, oh, ow), iterate over kernel positions.
    # Input position: id = (od + PD - kd) / SD if (od + PD - kd) % SD == 0
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

                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # Loop over input channels - accumulate sum_ic x[ic] * w[ic, oc]
                # x: [N, IC, ID, IH, IW]
                # w: [IC, OC, KD, KH, KW]
                for ic in range(0, IC):
                    x_offset = ((pid_n * IC + ic) * ID + id_) * IH * IW + ih * IW + iw  # [BLOCK_SP]
                    x_val = tl.load(x_ptr + x_offset, mask=spatial_valid, other=0.0)  # [BLOCK_SP]

                    w_offset = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw  # [BLOCK_OC]
                    w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_val[:, None] * x_val[None, :]

    # Add conv bias
    bc = tl.load(b_conv_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bc[:, None]

    # b_extra is per-output-channel: shape (OC,) flattened from (OC,1,1,1)
    be = tl.load(b_extra_ptr + oc_offs, mask=oc_mask, other=0.0)

    # original_x = acc (after conv+conv_bias)
    # y = (2*acc + be) * acc + acc
    y = (2.0 * acc + be[:, None]) * acc + acc

    # Write output: out[N, OC, OD, OH, OW]
    out_offset = ((pid_n * OC + oc_offs[:, None]) * SP_total) + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offset, y, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

        # Use a real ConvTranspose3d to get matching default init
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        w = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        b_conv = self.conv_transpose.bias.contiguous()  # [OC]
        b_extra = self.bias.contiguous().view(-1)  # [OC]

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OPD = OPH = OPW = self.output_padding

        OD = (ID - 1) * SD - 2 * PD + KD + OPD
        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        SP_total = OD * OH * OW
        grid = (
            N,
            (OC + BLOCK_OC - 1) // BLOCK_OC,
            (SP_total + BLOCK_SP - 1) // BLOCK_SP,
        )

        conv_transpose3d_kernel[grid](
            x, w, b_conv, b_extra, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out