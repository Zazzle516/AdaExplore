import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64,  'BLOCK_OC': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 64,  'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SP': 64,  'BLOCK_OC': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 32,  'BLOCK_OC': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SP': 64,  'BLOCK_OC': 32},  num_warps=4, num_stages=3),
    ],
    key=['IC', 'OC', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv_transpose3d_phase_kernel(
    x_ptr, w_ptr, b_conv_ptr, b_extra_ptr, out_ptr,
    N,
    IC: tl.constexpr, ID: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    PHASE_D: tl.constexpr, PHASE_H: tl.constexpr, PHASE_W: tl.constexpr,
    BLOCK_OC: tl.constexpr, BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    # sub-grid dimensions for this phase
    # output index od where (od + PD - kd) % SD == 0 for some kd with kd%SD == PHASE_D
    # We iterate sub-output grid; size = ceil((OD - phase_od_start) / SD)
    # phase_od_start: smallest od >= 0 such that (od + PD) % SD == PHASE_D
    # => od % SD == (PHASE_D - PD) % SD
    phase_od_start = (PHASE_D - PD) % SD
    phase_oh_start = (PHASE_H - PH) % SH
    phase_ow_start = (PHASE_W - PW) % SW

    SOD = (OD - phase_od_start + SD - 1) // SD
    SOH = (OH - phase_oh_start + SH - 1) // SH
    SOW = (OW - phase_ow_start + SW - 1) // SW
    SP_TOTAL = SOD * SOH * SOW

    if SP_TOTAL <= 0:
        return

    SOHW = SOH * SOW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    ic_range = tl.arange(0, IC)

    sp_mask = sp_offs < SP_TOTAL
    oc_mask = oc_offs < OC

    sub_od = sp_offs // SOHW
    rem = sp_offs % SOHW
    sub_oh = rem // SOW
    sub_ow = rem % SOW

    od = sub_od * SD + phase_od_start
    oh = sub_oh * SH + phase_oh_start
    ow = sub_ow * SW + phase_ow_start

    IHW = IH * IW
    IDHW = ID * IH * IW
    KDHW = KD * KH * KW
    OC_KDHW = OC * KDHW

    x_base = pid_n * (IC * IDHW)

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each kd with kd % SD == PHASE_D, kd in [0, KD)
    # number of such kd: depends on KD and SD
    # For KD=3, SD=2: phase 0 -> kd in {0,2}, phase 1 -> kd in {1}
    # We unroll over KD and check kd % SD == PHASE_D as constexpr.
    for kd in tl.static_range(0, KD):
        if (kd % SD) == PHASE_D:
            id_q = (od + PD - kd) // SD
            id_valid = (id_q >= 0) & (id_q < ID)
            for kh in tl.static_range(0, KH):
                if (kh % SH) == PHASE_H:
                    ih_q = (oh + PH - kh) // SH
                    ih_valid = (ih_q >= 0) & (ih_q < IH)
                    for kw in tl.static_range(0, KW):
                        if (kw % SW) == PHASE_W:
                            iw_q = (ow + PW - kw) // SW
                            iw_valid = (iw_q >= 0) & (iw_q < IW)
                            spatial_valid = id_valid & ih_valid & iw_valid & sp_mask

                            in_spatial = id_q * IHW + ih_q * IW + iw_q

                            x_off = x_base + ic_range[None, :] * IDHW + in_spatial[:, None]
                            x_tile = tl.load(x_ptr + x_off, mask=spatial_valid[:, None], other=0.0)

                            khw_off = kd * (KH * KW) + kh * KW + kw
                            w_off = ic_range[:, None] * OC_KDHW + oc_offs[None, :] * KDHW + khw_off
                            w_tile = tl.load(w_ptr + w_off, mask=oc_mask[None, :], other=0.0)

                            acc += tl.dot(x_tile, w_tile, out_dtype=tl.float32)

    bc = tl.load(b_conv_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bc[None, :]

    be = tl.load(b_extra_ptr + oc_offs, mask=oc_mask, other=0.0)

    y = (2.0 * acc + be[None, :]) * acc + acc

    OHW = OH * OW
    SP_TOTAL_OUT = OD * OH * OW
    out_spatial = od * OHW + oh * OW + ow
    out_off = (pid_n * OC * SP_TOTAL_OUT
               + oc_offs[None, :] * SP_TOTAL_OUT
               + out_spatial[:, None])
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

        weight = self.conv_transpose.weight.contiguous()
        b_conv = self.conv_transpose.bias.contiguous()
        b_extra = self.bias.contiguous().view(-1)

        for PHASE_D in range(SD):
            for PHASE_H in range(SH):
                for PHASE_W in range(SW):
                    phase_od_start = (PHASE_D - PD) % SD
                    phase_oh_start = (PHASE_H - PH) % SH
                    phase_ow_start = (PHASE_W - PW) % SW
                    SOD = (OD - phase_od_start + SD - 1) // SD
                    SOH = (OH - phase_oh_start + SH - 1) // SH
                    SOW = (OW - phase_ow_start + SW - 1) // SW
                    SP_TOTAL = SOD * SOH * SOW
                    if SP_TOTAL <= 0:
                        continue

                    grid = lambda META, sp=SP_TOTAL: (
                        N,
                        (OC + META['BLOCK_OC'] - 1) // META['BLOCK_OC'],
                        (sp + META['BLOCK_SP'] - 1) // META['BLOCK_SP'],
                    )

                    conv_transpose3d_phase_kernel[grid](
                        x, weight, b_conv, b_extra, out,
                        N, IC, ID, IH, IW,
                        OC, OD, OH, OW,
                        KD, KH, KW,
                        SD, SH, SW,
                        PD, PH, PW,
                        PHASE_D, PHASE_H, PHASE_W,
                    )
        return out