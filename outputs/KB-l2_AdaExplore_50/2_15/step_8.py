import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_bn_kernel(
    x_ptr, w_ptr, scale_ptr, shift_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
):
    # one program per (n, oc_tile, spatial_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    OUT_SPATIAL = OD * OH * OW

    sp_offs = pid_sp * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
    sp_mask = sp_offs < OUT_SPATIAL

    od = sp_offs // (OH * OW)
    rem = sp_offs - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # accumulator [BLOCK_OC, BLOCK_SPATIAL]
    acc = tl.zeros((BLOCK_OC, BLOCK_SPATIAL), dtype=tl.float32)

    # For each output position (od, oh, ow):
    #   output[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw) of:
    #     input[n, ic, id, ih, iw] * weight[ic, oc, kd, kh, kw]
    # where id*SD - PD + kd = od  =>  id = (od + PD - kd) / SD  (must be integer)
    # We iterate over kd, kh, kw, ic and accumulate.

    # Pre-compute (od+PD), (oh+PH), (ow+PW)
    od_p = od + PD
    oh_p = oh + PH
    ow_p = ow + PW

    for kd in range(0, KD):
        id_num = od_p - kd
        id_q = id_num // SD
        id_r = id_num - id_q * SD
        valid_d = (id_r == 0) & (id_q >= 0) & (id_q < ID)
        for kh in range(0, KH):
            ih_num = oh_p - kh
            ih_q = ih_num // SH
            ih_r = ih_num - ih_q * SH
            valid_h = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)
            for kw in range(0, KW):
                iw_num = ow_p - kw
                iw_q = iw_num // SW
                iw_r = iw_num - iw_q * SW
                valid_w = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW)
                valid = valid_d & valid_h & valid_w  # [BLOCK_SPATIAL]

                # input index base: x[n, ic, id_q, ih_q, iw_q]
                # x stride: (IC*ID*IH*IW, ID*IH*IW, IH*IW, IW, 1)
                in_spatial = id_q * (IH * IW) + ih_q * IW + iw_q  # [BLOCK_SPATIAL]

                # weight[ic, oc, kd, kh, kw]
                # weight stride: (OC*KD*KH*KW, KD*KH*KW, KH*KW, KW, 1)
                w_kernel_off = kd * (KH * KW) + kh * KW + kw

                for ic in range(0, IC):
                    x_off = (pid_n * IC + ic) * (ID * IH * IW) + in_spatial
                    x_vals = tl.load(x_ptr + x_off, mask=sp_mask & valid, other=0.0)  # [BLOCK_SPATIAL]

                    w_off = ic * (OC * KD * KH * KW) + oc_offs * (KD * KH * KW) + w_kernel_off
                    w_vals = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    acc += w_vals[:, None] * x_vals[None, :]

    # Apply BN affine: out = acc * scale + shift
    scale = tl.load(scale_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    shift = tl.load(shift_ptr + oc_offs, mask=oc_mask, other=0.0)
    out = acc * scale[:, None] + shift[:, None]

    # store output[n, oc, od, oh, ow]
    out_off = (pid_n * OC + oc_offs[:, None]) * OUT_SPATIAL + sp_offs[None, :]
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


@triton.jit
def sub_mean_kernel(
    x_ptr, out_ptr,
    S,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * S

    sum_val = tl.zeros((), dtype=tl.float32)
    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        sum_val += tl.sum(x, axis=0)
    mean = sum_val / S

    for off in range(0, S, BLOCK_SIZE):
        offs = off + tl.arange(0, BLOCK_SIZE)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs, mask=mask, other=0.0)
        tl.store(out_ptr + row_start + offs, x - mean, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

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

    def forward(self, x):
        if self.training:
            # fallback to PyTorch for training (BN stats update)
            x = self.conv_transpose(x)
            x = self.batch_norm(x)
            x = x - torch.mean(x, dim=(2, 3, 4), keepdim=True)
            return x

        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        SD, SH, SW = self.sd, self.sh, self.sw
        PD, PH, PW = self.pd, self.ph, self.pw

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        # Fold BN affine into output: y = (conv_out + conv_bias - running_mean) / sqrt(var+eps) * gamma + beta
        # => y = conv_out * scale + shift
        # where scale = gamma / sqrt(var+eps)
        #       shift = (conv_bias - running_mean) * scale + beta
        bn = self.batch_norm
        eps = bn.eps
        gamma = bn.weight
        beta = bn.bias
        rm = bn.running_mean
        rv = bn.running_var
        scale = gamma / torch.sqrt(rv + eps)  # [OC]
        conv_bias = self.conv_transpose.bias if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        shift = (conv_bias - rm) * scale + beta  # [OC]

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        scale = scale.contiguous()
        shift = shift.contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        OUT_SPATIAL = OD * OH * OW
        BLOCK_OC = 32
        BLOCK_SPATIAL = 64

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OUT_SPATIAL, BLOCK_SPATIAL))
        conv_transpose3d_bn_kernel[grid](
            x, weight, scale, shift, out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_SPATIAL=BLOCK_SPATIAL,
            num_warps=4,
            num_stages=2,
        )

        # subtract mean along spatial dims
        S = OD * OH * OW
        out_flat = out.view(N * OC, S)
        out2 = torch.empty_like(out_flat)
        sub_mean_kernel[(N * OC,)](out_flat, out2, S, BLOCK_SIZE=1024, num_warps=4)
        return out2.view(N, OC, OD, OH, OW)