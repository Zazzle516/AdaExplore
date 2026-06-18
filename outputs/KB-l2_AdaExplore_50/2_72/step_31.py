import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Scatter-add ConvTranspose3d kernel.
# One program per (n, ic, id) -- loops over (ih, iw) inside.
# For each input element, scatters its contribution to (kd, kh, kw, oc) output positions.
@triton.jit
def conv_transpose3d_scatter_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_ic = tl.program_id(1)
    pid_id = tl.program_id(2)

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Preload weight for this ic across all (oc, kd, kh, kw)
    # weight shape: (IC, OC, KD, KH, KW)
    # We'll load per (kd,kh,kw) inside the loop.

    for ih in range(IH):
        for iw in range(IW):
            # load x[n, ic, id, ih, iw]
            x_off = (((pid_n * IC + pid_ic) * ID + pid_id) * IH + ih) * IW + iw
            x_val = tl.load(x_ptr + x_off)

            for kd in tl.static_range(KD):
                od = pid_id * SD - PD + kd
                od_valid = (od >= 0) & (od < OD)
                for kh in tl.static_range(KH):
                    oh = ih * SH - PH + kh
                    oh_valid = (oh >= 0) & (oh < OH)
                    for kw in tl.static_range(KW):
                        ow = iw * SW - PW + kw
                        ow_valid = (ow >= 0) & (ow < OW)
                        valid = od_valid & oh_valid & ow_valid

                        # weight offset for all OC at this (ic, kd, kh, kw)
                        w_off = (((pid_ic * OC + oc_offs) * KD + kd) * KH + kh) * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                        contrib = x_val * w_val

                        out_off = (((pid_n * OC + oc_offs) * OD + od) * OH + oh) * OW + ow
                        tl.atomic_add(out_ptr + out_off, contrib, mask=oc_mask & valid)


@triton.jit
def bn_avgpool4_kernel(
    in_ptr, bias_ptr, scale_ptr, shift_ptr, out_ptr,
    N, C, ID, IH, IW,
    OD, OH, OW,
    BLOCK: tl.constexpr,
):
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
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)

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
    # add bias (since conv_out doesn't include bias)
    acc = acc + bias
    # BN
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

        # Initialize conv_out to zeros (we will scatter-add into it; bias added in epilogue)
        conv_out = torch.zeros((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        grid = (N, IC, ID)
        conv_transpose3d_scatter_kernel[grid](
            x, weight, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        if self.training:
            x2 = conv_out + bias.view(1, -1, 1, 1, 1)
            x2 = self.batch_norm(x2)
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

        OD2 = OD // 4
        OH2 = OH // 4
        OW2 = OW // 4

        if OD2 * 4 != OD or OH2 * 4 != OH or OW2 * 4 != OW:
            x2 = conv_out + bias.view(1, -1, 1, 1, 1)
            x2 = x2 * scale.view(1, -1, 1, 1, 1) + shift.view(1, -1, 1, 1, 1)
            x2 = F.avg_pool3d(x2, 2)
            x2 = F.avg_pool3d(x2, 2)
            return x2

        out = torch.empty((N, OC, OD2, OH2, OW2), device=x.device, dtype=x.dtype)
        total = N * OC * OD2 * OH2 * OW2
        BLOCK = 256
        grid2 = ((total + BLOCK - 1) // BLOCK,)
        bn_avgpool4_kernel[grid2](
            conv_out, bias, scale, shift, out,
            N, OC, OD, OH, OW,
            OD2, OH2, OW2,
            BLOCK=BLOCK,
            num_warps=4,
        )

        return out