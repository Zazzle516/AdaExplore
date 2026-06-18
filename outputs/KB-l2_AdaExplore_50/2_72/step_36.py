import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy:
# - Compute conv_transpose3d output directly into a (N, OC, OD, OH, OW) buffer with a
#   gather kernel (one program per (n, od, oh*ow), all OC computed at once).
# - Then fuse BN + avgpool(4) into a single epilogue kernel.
#
# Key optimization: BLOCK over OW dimension so each program produces a vector of OW outputs,
# improving memory coalescing and reducing kernel launch overhead. With IC=3, KD=KH=KW=3,
# the inner loop has 81 ops, modest enough that we benefit from vectorizing across OW.


@triton.jit
def conv_transpose3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_hw = tl.program_id(2)

    # pid_hw indexes (oh, ow_block)
    n_ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    oh = pid_hw // n_ow_blocks
    ow_blk = pid_hw % n_ow_blocks

    ow_offs = ow_blk * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    ow_mask = ow_offs < OW

    oc_offs = tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    # acc shape: [BLOCK_OC, BLOCK_OW]
    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    # output[n, oc, pid_d, oh, ow] = sum over (ic, kd, kh, kw) of
    #   x[n, ic, id, ih, iw] * w[ic, oc, kd, kh, kw]
    # where id = (pid_d + PD - kd) / SD must be integer in [0, ID)
    #       ih = (oh    + PH - kh) / SH must be integer in [0, IH)
    #       iw = (ow    + PW - kw) / SW must be integer in [0, IW)

    for kd in tl.static_range(KD):
        id_num = pid_d + PD - kd
        id_val = id_num // SD
        id_valid = (id_num >= 0) & ((id_num - id_val * SD) == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(KH):
            ih_num = oh + PH - kh
            ih_val = ih_num // SH
            ih_valid = (ih_num >= 0) & ((ih_num - ih_val * SH) == 0) & (ih_val >= 0) & (ih_val < IH)
            for kw in tl.static_range(KW):
                # per-ow validity: iw_num = ow + PW - kw, must be divisible by SW
                iw_num = ow_offs + PW - kw  # [BLOCK_OW]
                iw_val = iw_num // SW
                iw_valid = (iw_num >= 0) & ((iw_num - iw_val * SW) == 0) & (iw_val >= 0) & (iw_val < IW) & ow_mask

                valid_2d = id_valid & ih_valid  # scalar
                full_mask = iw_valid & valid_2d  # [BLOCK_OW]

                for ic in tl.static_range(IC):
                    # Load x[n, ic, id_val, ih_val, iw_val] for each ow in block
                    x_off = ((pid_n * IC + ic) * ID + id_val) * IH * IW + ih_val * IW + iw_val
                    x_val = tl.load(x_ptr + x_off, mask=full_mask, other=0.0)  # [BLOCK_OW]

                    # Load w[ic, oc, kd, kh, kw] for all oc
                    w_off = ((ic * OC + oc_offs) * KD + kd) * KH * KW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                    # outer product into acc
                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    acc += b_val[:, None]

    # Store: out[n, oc, pid_d, oh, ow]
    out_off = ((pid_n * OC + oc_offs[:, None]) * OD + pid_d) * OH * OW + oh * OW + ow_offs[None, :]
    store_mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


@triton.jit
def bn_avgpool4_kernel(
    in_ptr, scale_ptr, shift_ptr, out_ptr,
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

        if self.training:
            x_ = self.conv_transpose(x)
            x_ = self.batch_norm(x_)
            x_ = F.avg_pool3d(x_, 2)
            x_ = F.avg_pool3d(x_, 2)
            return x_

        # Power of 2 >= OC
        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        # Choose BLOCK_OW: tile OW dimension. OW = 63 typically.
        BLOCK_OW = 64

        conv_out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        n_ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
        grid = (N, OD, OH * n_ow_blocks)

        conv_transpose3d_kernel[grid](
            x, weight, bias, conv_out,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_OW=BLOCK_OW,
            num_warps=4,
            num_stages=2,
        )

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

        if OD2 == 0 or OH2 == 0 or OW2 == 0:
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