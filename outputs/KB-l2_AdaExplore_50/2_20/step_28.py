import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_gather_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [IC, OC, KD, KH, KW]
    cbias_ptr,    # [OC] - conv transpose bias
    ebias_ptr,    # [OC] - epilogue bias
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    sp_mask = sp_offs < ODHW

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each output position (od, oh, ow):
    # output[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw) of
    #   input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where id*STRIDE_D = od + PAD_D - kd, similarly for h, w
    # so id_num = od + PAD_D - kd, must be divisible by STRIDE_D, and in [0, ID)

    for kd in tl.static_range(KD):
        id_num = od + PAD_D - kd  # [BLOCK_SP]
        id_val = id_num // STRIDE_D
        id_valid = (id_num % STRIDE_D == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(KH):
            ih_num = oh + PAD_H - kh
            ih_val = ih_num // STRIDE_H
            ih_valid = (ih_num % STRIDE_H == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(KW):
                iw_num = ow + PAD_W - kw
                iw_val = iw_num // STRIDE_W
                iw_valid = (iw_num % STRIDE_W == 0) & (iw_val >= 0) & (iw_val < IW)
                spatial_valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # input offset for [n, :, id_val, ih_val, iw_val]
                # input layout: [N, IC, ID, IH, IW]
                in_spatial_off = id_val * (IH * IW) + ih_val * IW + iw_val  # [BLOCK_SP]
                in_base = pid_n * (IC * ID * IH * IW) + in_spatial_off  # [BLOCK_SP]

                # weight offset for [:, oc_offs, kd, kh, kw]
                # weight layout: [IC, OC, KD, KH, KW]
                w_kspatial = kd * (KH * KW) + kh * KW + kw
                w_base = oc_offs * (KD * KH * KW) + w_kspatial  # [BLOCK_OC]

                # Loop over IC
                for ic in range(IC):
                    in_ptrs = in_base + ic * (ID * IH * IW)  # [BLOCK_SP]
                    w_ptrs = w_base + ic * (OC * KD * KH * KW)  # [BLOCK_OC]
                    x_vals = tl.load(x_ptr + in_ptrs, mask=spatial_valid, other=0.0)  # [BLOCK_SP]
                    w_vals = tl.load(w_ptr + w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += x_vals[:, None] * w_vals[None, :]

    # Add conv bias
    cbias = tl.load(cbias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc = acc + cbias[None, :]

    # Add epilogue bias and apply (2x + b) * x + x
    ebias = tl.load(ebias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    out = (2.0 * acc + ebias[None, :]) * acc + acc

    # Write output [N, OC, OD, OH, OW]
    out_base = pid_n * (OC * ODHW) + oc_offs[None, :] * ODHW + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_base, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        if isinstance(stride, int):
            self.sd = self.sh = self.sw = stride
        else:
            self.sd, self.sh, self.sw = stride
        if isinstance(padding, int):
            self.pd = self.ph = self.pw = padding
        else:
            self.pd, self.ph, self.pw = padding
        if isinstance(output_padding, int):
            self.opd = self.oph = self.opw = output_padding
        else:
            self.opd, self.oph, self.opw = output_padding

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        SD, SH, SW = self.sd, self.sh, self.sw
        PD, PH, PW = self.pd, self.ph, self.pw
        OPD, OPH, OPW = self.opd, self.oph, self.opw

        OD = (ID - 1) * SD - 2 * PD + KD + OPD
        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        cbias = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        ebias = self.bias.contiguous().view(-1)

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 64

        ODHW = OD * OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (ODHW + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose3d_gather_kernel[grid](
            x, weight, cbias, ebias, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC, BLOCK_SP=BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out