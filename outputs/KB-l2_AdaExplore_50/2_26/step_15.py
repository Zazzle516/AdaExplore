import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_fused_kernel(
    x_ptr, w_ptr, bias_ptr, add_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_total = OD * OH * OW
    sp_mask = sp_offs < sp_total

    od = sp_offs // (OH * OW)
    rem = sp_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    # Accumulator
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # For each output coord, sum over (ic, kd, kh, kw) where input coord is valid
    # id_unstrided = od + PD - kd  must be divisible by SD
    # id = id_unstrided / SD, in [0, ID)
    for kd in tl.static_range(0, KD):
        id_un = od + PD - kd
        id_ok = (id_un % SD == 0)
        id_v = id_un // SD
        id_in = (id_v >= 0) & (id_v < ID) & id_ok
        for kh in tl.static_range(0, KH):
            ih_un = oh + PH - kh
            ih_ok = (ih_un % SH == 0)
            ih_v = ih_un // SH
            ih_in = (ih_v >= 0) & (ih_v < IH) & ih_ok
            for kw in tl.static_range(0, KW):
                iw_un = ow + PW - kw
                iw_ok = (iw_un % SW == 0)
                iw_v = iw_un // SW
                iw_in = (iw_v >= 0) & (iw_v < IW) & iw_ok

                valid = id_in & ih_in & iw_in  # [BLOCK_SP]

                # input offset base for this spatial pos (per ic stride)
                # x layout: [N, IC, ID, IH, IW]
                in_spatial = id_v * (IH * IW) + ih_v * IW + iw_v  # [BLOCK_SP]
                x_base = pid_n * (IC * ID * IH * IW) + in_spatial  # [BLOCK_SP]

                # weight offset for this (kd,kh,kw): w[ic, oc, kd, kh, kw]
                # w layout: [IC, OC, KD, KH, KW]
                w_khw = kd * (KH * KW) + kh * KW + kw

                # Loop over IC
                for ic in range(0, IC):
                    x_ptrs = x_ptr + x_base + ic * (ID * IH * IW)
                    x_vals = tl.load(x_ptrs, mask=valid & sp_mask, other=0.0)  # [BLOCK_SP]

                    w_ptrs = w_ptr + ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_khw
                    w_vals = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += x_vals[:, None] * w_vals[None, :]

    # Add conv bias
    b = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += b[None, :]

    # Add add_input: layout [N, OC, OD, OH, OW]
    add_base = pid_n * (OC * OD * OH * OW) + oc_offs[None, :] * (OD * OH * OW) + sp_offs[:, None]
    add_mask = oc_mask[None, :] & sp_mask[:, None]
    add_vals = tl.load(add_ptr + add_base, mask=add_mask, other=0.0)
    v = acc + add_vals

    # hardswish: v * v * relu6(v+3)/6
    t = v + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    out = v * v * t * (1.0 / 6.0)

    tl.store(out_ptr + add_base, out, mask=add_mask)


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

    def forward(self, x, add_input):
        x = x.contiguous()
        add_input = add_input.contiguous()
        w = self.conv_transpose.weight.contiguous()
        cbias = self.conv_transpose.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding

        OD = (ID - 1) * SD - 2 * PD + KD + self.output_padding
        OH = (IH - 1) * SH - 2 * PH + KH + self.output_padding
        OW = (IW - 1) * SW - 2 * PW + KW + self.output_padding

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 32
        BLOCK_SP = 128

        sp_total = OD * OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (sp_total + BLOCK_SP - 1) // BLOCK_SP)

        conv_transpose3d_fused_kernel[grid](
            x, w, cbias, add_input, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC, BLOCK_SP,
            num_warps=4, num_stages=2,
        )
        return out