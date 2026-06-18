import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose_fused_kernel(
    x_ptr,        # [N, IC, IH, IW]
    w_ptr,        # [IC, OC, KH, KW]
    b_conv_ptr,   # [OC]
    b_extra_ptr,  # [OC]
    out_ptr,      # [N, OC, OH, OW]
    N, IC, OC,
    IH, IW, OH, OW,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    inv_scale,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # program ids: (n, oc_block, sp_block)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offsets = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offsets < OC
    sp_mask = sp_offsets < (OH * OW)

    oh = sp_offsets // OW
    ow = sp_offsets % OW

    # accumulator
    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    # for each output position (oh, ow), iterate over (ic, kh, kw)
    # input position: ih = (oh + PAD_H - kh) / STRIDE_H if divisible
    #                 iw = (ow + PAD_W - kw) / STRIDE_W if divisible
    for kh in range(KH):
        oh_pad_kh = oh + PAD_H - kh  # [BLOCK_SP]
        ih = oh_pad_kh // STRIDE_H
        ih_valid = (oh_pad_kh % STRIDE_H == 0) & (ih >= 0) & (ih < IH)

        for kw in range(KW):
            ow_pad_kw = ow + PAD_W - kw
            iw = ow_pad_kw // STRIDE_W
            iw_valid = (ow_pad_kw % STRIDE_W == 0) & (iw >= 0) & (iw < IW)

            spatial_valid = ih_valid & iw_valid & sp_mask  # [BLOCK_SP]

            for ic in range(IC):
                # load x[n, ic, ih, iw] -> [BLOCK_SP]
                x_idx = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + x_idx, mask=spatial_valid, other=0.0)

                # load w[ic, oc, kh, kw] -> [BLOCK_OC]
                w_idx = ic * OC * KH * KW + oc_offsets * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)

                # outer product accumulate
                acc += x_val[:, None] * w_val[None, :]

    # add conv bias
    bc = tl.load(b_conv_ptr + oc_offsets, mask=oc_mask, other=0.0)
    be = tl.load(b_extra_ptr + oc_offsets, mask=oc_mask, other=0.0)
    acc = acc + bc[None, :] + be[None, :]

    # clamp [0,1], *scale, clamp [0,1], /scale
    # equivalent: min(scale * clamp(x,0,1), 1) * inv_scale
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * (1.0 / inv_scale)  # scaling_factor
    # wait: inv_scale = 1/scaling_factor; scale*x => x/inv_scale
    # Let me redo: pass scaling_factor instead.
    # We'll use inv_scale as scaling_factor here for simplicity rename below.
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * inv_scale  # divide by scaling_factor = multiply by inv_scale

    # store output
    out_idx = (pid_n * OC * OH * OW
               + oc_offsets[None, :] * OH * OW
               + sp_offsets[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.jit
def conv_transpose_fused_kernel_v2(
    x_ptr,
    w_ptr,
    b_conv_ptr,
    b_extra_ptr,
    out_ptr,
    N, IC, OC,
    IH, IW, OH, OW,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    scaling_factor, inv_scale,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offsets = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offsets = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offsets < OC
    sp_mask = sp_offsets < (OH * OW)

    oh = sp_offsets // OW
    ow = sp_offsets % OW

    acc = tl.zeros((BLOCK_SP, BLOCK_OC), dtype=tl.float32)

    for kh in range(KH):
        oh_pad_kh = oh + PAD_H - kh
        ih = oh_pad_kh // STRIDE_H
        ih_valid = (oh_pad_kh % STRIDE_H == 0) & (ih >= 0) & (ih < IH)

        for kw in range(KW):
            ow_pad_kw = ow + PAD_W - kw
            iw = ow_pad_kw // STRIDE_W
            iw_valid = (ow_pad_kw % STRIDE_W == 0) & (iw >= 0) & (iw < IW)

            spatial_valid = ih_valid & iw_valid & sp_mask

            for ic in range(IC):
                x_idx = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_val = tl.load(x_ptr + x_idx, mask=spatial_valid, other=0.0)

                w_idx = ic * OC * KH * KW + oc_offsets * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)

                acc += x_val[:, None] * w_val[None, :]

    bc = tl.load(b_conv_ptr + oc_offsets, mask=oc_mask, other=0.0)
    be = tl.load(b_extra_ptr + oc_offsets, mask=oc_mask, other=0.0)
    acc = acc + bc[None, :] + be[None, :]

    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * scaling_factor
    acc = tl.minimum(tl.maximum(acc, 0.0), 1.0)
    acc = acc * inv_scale

    out_idx = (pid_n * OC * OH * OW
               + oc_offsets[None, :] * OH * OW
               + sp_offsets[:, None])
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = float(scaling_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous()  # [IC, OC, KH, KW]
        b_conv = self.conv_transpose.bias.contiguous()  # [OC]
        b_extra = self.bias.view(-1).contiguous()  # [OC]

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        SH = SW = self.stride
        PH = PW = self.padding
        OPH = OPW = self.output_padding

        OH = (IH - 1) * SH - 2 * PH + KH + OPH
        OW = (IW - 1) * SW - 2 * PW + KW + OPW

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_SP = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_SP))

        conv_transpose_fused_kernel_v2[grid](
            x, w, b_conv, b_extra, out,
            N, IC, OC,
            IH, IW, OH, OW,
            KH, KW,
            SH, SW,
            PH, PW,
            self.scaling_factor, 1.0 / self.scaling_factor,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SP=BLOCK_SP,
            num_warps=4,
            num_stages=2,
        )

        return out