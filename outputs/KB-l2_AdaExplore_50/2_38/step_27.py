import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, OC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    # grid: (N, ceil(OD*OH*OW / BLOCK_SP), ceil(OC / BLOCK_OC))
    n = tl.program_id(0)
    sp_block = tl.program_id(1)
    oc_block = tl.program_id(2)

    sp_offs = sp_block * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)

    OHW = OH * OW
    OS = OD * OHW

    sp_mask = sp_offs < OS
    oc_mask = oc_offs < OC

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    # for each output position, find which (kd, kh, kw, ic) contribute
    # output[od, oh, ow] = sum over (ic, kd, kh, kw):
    #   x[ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where: od + PD = id * SD + kd  =>  id*SD = od + PD - kd  => id = (od+PD-kd)/SD if divisible
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # Precompute base offsets
    # x layout: [N, IC, ID, IH, IW]
    # w layout: [IC, OC, KD, KH, KW]
    x_n_base = n * IC * ID * IH * IW

    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_ = id_num // SD
        id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_ >= 0) & (id_ < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_ = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_ >= 0) & (ih_ < IH)
            for kw in tl.static_range(0, KW):
                iw_num = ow + PW - kw
                iw_ = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_ >= 0) & (iw_ < IW)
                valid = id_valid & ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

                # input position offset within batch (ic varies)
                in_spatial = id_ * IH * IW + ih_ * IW + iw_  # [BLOCK_SP]
                # weight position offset within (ic, oc): kd*KH*KW + kh*KW + kw
                w_k_off = kd * KH * KW + kh * KW + kw

                # Loop over ic
                for ic in range(0, IC):
                    x_off = x_n_base + ic * (ID * IH * IW) + in_spatial  # [BLOCK_SP]
                    xv = tl.load(x_ptr + x_off, mask=valid, other=0.0)  # [BLOCK_SP]
                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_k_off  # [BLOCK_OC]
                    wv = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += xv[:, None] * wv[None, :]

    # add bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += bias[None, :]

    # write output: out layout [N, OC, OD, OH, OW]
    out_off = n * OC * OS + oc_offs[None, :] * OS + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


@triton.jit
def _clamp_softmax_scale_kernel(
    x_ptr, scale_ptr, out_ptr,
    S, C,
    clamp_min, clamp_max,
    BLOCK: tl.constexpr,
):
    bc = tl.program_id(0)
    c = bc % C
    row_start = bc * S

    max_val = -float('inf')
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=-float('inf'))
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        v = tl.where(mask, v, -float('inf'))
        block_max = tl.max(v, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_val = 0.0
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - max_val)
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    inv_sum = 1.0 / sum_val
    s = tl.load(scale_ptr + c)

    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < S
        v = tl.load(x_ptr + row_start + idx, mask=mask, other=0.0)
        v = tl.minimum(tl.maximum(v, clamp_min), clamp_max)
        e = tl.exp(v - max_val) * inv_sum * s
        tl.store(out_ptr + row_start + idx, e, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        self.stride = stride if isinstance(stride, tuple) else (stride,) * 3
        self.padding = padding if isinstance(padding, tuple) else (padding,) * 3
        self.output_padding = output_padding if isinstance(output_padding, tuple) else (output_padding,) * 3
        self.pool_kernel_size = pool_kernel_size
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # Mirror nn.ConvTranspose3d params
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1, 1))

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.contiguous()

        N, IC, ID, IH, IW = x.shape
        KD, KH, KW = self.kernel_size
        SD, SH, SW = self.stride
        PD, PH, PW = self.padding
        OPD, OPH, OPW = self.output_padding

        OD = (ID - 1) * SD - 2 * PD + KD + OPD
        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        OC = self.out_channels
        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        bias = self.conv_transpose.bias.contiguous()

        out_conv = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        OS = OD * OH * OW
        BLOCK_SP = 64
        BLOCK_OC = 32

        grid = (N, (OS + BLOCK_SP - 1) // BLOCK_SP, (OC + BLOCK_OC - 1) // BLOCK_OC)
        _conv_transpose3d_kernel[grid](
            x, weight, bias, out_conv,
            N, IC, OC,
            ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_SP=BLOCK_SP,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
            num_stages=2,
        )

        # Now do clamp + softmax + scale
        S = OD * OH * OW
        x_flat = out_conv.view(N * OC, S)
        out = torch.empty_like(x_flat)
        scale_flat = self.scale.view(-1).contiguous()

        if S <= 1024:
            BLOCK = triton.next_power_of_2(S)
            num_warps = 4 if BLOCK <= 256 else 8
        else:
            BLOCK = 1024
            num_warps = 8

        grid2 = (N * OC,)
        _clamp_softmax_scale_kernel[grid2](
            x_flat, scale_flat, out,
            S, OC,
            self.clamp_min, self.clamp_max,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )
        return out.view(N, OC, OD, OH, OW)