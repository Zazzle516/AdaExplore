import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
):
    # program: (n, od, oh*ow) -- one program per (n, od, spatial element), loop over OC tile
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oh = pid_hw // OW
    ow = pid_hw % OW

    # iterate OC in tiles
    for oc_start in tl.static_range(0, 1):
        pass

    oc_offs = tl.arange(0, BLOCK_OC)
    # We assume OC <= BLOCK_OC (16 in our case)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    # For ConvTranspose3d:
    # output[n, oc, od, oh, ow] = sum over (ic, kd, kh, kw) of
    #   x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id*SD - PD + kd = od  =>  id = (od + PD - kd) / SD, must be integer and in [0, ID)
    
    for kd in range(KD):
        id_num = pid_d + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & (id_num % SD == 0) & (id_val >= 0) & (id_val < ID)
        for kh in range(KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & (ih_num % SH == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in range(KW):
                iw_num = ow + PW - kw
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & (iw_num % SW == 0) & (iw_val >= 0) & (iw_val < IW)
                valid = id_valid & ih_valid & iw_valid
                # loop over ic
                for ic in range(IC):
                    x_off = ((pid_n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                    x_val = tl.load(x_ptr + x_off, mask=valid, other=0.0)
                    # weight shape: (IC, OC, KD, KH, KW)
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += x_val * w_val

    # add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val

    # store
    out_off = ((pid_n * OC + oc_offs) * OD + pid_d) * OH * OW + oh * OW + ow
    tl.store(out_ptr + out_off, acc, mask=oc_mask)


@triton.jit
def bn_avgpool4_kernel(
    in_ptr, scale_ptr, shift_ptr, out_ptr,
    N, C, ID, IH, IW,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
    # one program per output element; output spatial = ID//4, IH//4, IW//4
    # output index: flat over N*C*OD*OH*OW
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * OD * OH * OW
    mask = offs < total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    scale = tl.load(scale_ptr + c, mask=mask, other=0.0)
    shift = tl.load(shift_ptr + c, mask=mask, other=0.0)

    acc = tl.zeros((BLOCK,), dtype=tl.float32)

    id_base = od * 4
    ih_base = oh * 4
    iw_base = ow * 4

    for dd in tl.static_range(4):
        for hh in tl.static_range(4):
            for ww in tl.static_range(4):
                id_v = id_base + dd
                ih_v = ih_base + hh
                iw_v = iw_base + ww
                in_off = ((n * C + c) * ID + id_v) * IH * IW + ih_v * IW + iw_v
                v = tl.load(in_ptr + in_off, mask=mask, other=0.0)
                acc += v

    acc = acc * (1.0 / 64.0)
    # apply BN: (x - mean) / sqrt(var+eps) * gamma + beta = x*scale + shift
    # Since avg is linear: avg(x*scale + shift) = avg(x)*scale + shift
    out = acc * scale + shift

    tl.store(out_ptr + offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv_transpose.weight.contiguous()
        bias = self.conv_transpose.bias.contiguous()

        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        # Choose BLOCK_OC to next power of 2 >= OC
        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        grid = (N, OD, OH * OW)
        conv_transpose3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        # BN fused with two avgpool(2) -> equivalent to avgpool(4)
        # Compute BN scale/shift in eval mode using running stats
        if self.training:
            # Fall back to torch in training mode
            x2 = self.batch_norm(conv_out)
            x2 = F.avg_pool3d(x2, 2)
            x2 = F.avg_pool3d(x2, 2)
            return x2

        running_mean = self.batch_norm.running_mean
        running_var = self.batch_norm.running_var
        gamma = self.batch_norm.weight
        beta = self.batch_norm.bias
        eps = self.batch_norm.eps

        inv_std = torch.rsqrt(running_var + eps)
        scale = (gamma * inv_std).contiguous()
        shift = (beta - running_mean * scale).contiguous()

        # avg pool over 4x4x4 blocks
        OD2 = OD // 4
        OH2 = OH // 4
        OW2 = OW // 4

        # If dims not divisible by 4, fall back
        if OD2 * 4 != OD or OH2 * 4 != OH or OW2 * 4 != OW:
            x2 = conv_out * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
            x2 = F.avg_pool3d(x2, 2)
            x2 = F.avg_pool3d(x2, 2)
            return x2

        out = torch.empty((N, OC, OD2, OH2, OW2), device=x.device, dtype=x.dtype)
        total = N * OC * OD2 * OH2 * OW2
        BLOCK = 256
        grid2 = ((total + BLOCK - 1) // BLOCK,)
        bn_avgpool4_kernel[grid2](
            conv_out, scale, shift, out,
            N, OC, OD, OH, OW,
            OD2, OH2, OW2,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out