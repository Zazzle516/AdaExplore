import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def convt3d_scatter_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    SD, SH, SW,
    PD, PH, PW,
    BLOCK_OC: tl.constexpr,
    BLOCK_IW: tl.constexpr,
):
    # grid: (N * ID * IH, IW_tiles, OC_tiles * KD * KH * KW)
    # actually simpler: one program per (n, id, ih, iw_tile, oc_tile, kd, kh, kw)
    # Let's restructure.
    pass


# Better approach: compute conv output via gather (one program per output element tile over OC)
# Output shape: (N, OC, OD, OH, OW). OD = (ID-1)*SD - 2*PD + KD, etc.
# For output position (od, oh, ow), the contributing input positions are:
#   id such that id*SD - PD + kd = od  =>  id = (od + PD - kd) / SD  with (od+PD-kd) % SD == 0
# Gather is easier.

@triton.jit
def fused_convt_lse_kernel(
    x_ptr,           # [N, IC, ID, IH, IW]
    w_ptr,           # [IC, OC, KD, KH, KW]
    cb_ptr,          # [OC] conv bias
    sb_ptr,          # scalar bias
    out_ptr,         # [N, 1, OD, OH, OW]
    N, IC,
    ID, IH, IW,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    SD: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh, ow_tile)
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)

    ow_start = pid_w * BLOCK_W
    ow_offs = ow_start + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    # decode pid -> n, od, oh
    oh = pid % OH
    tmp = pid // OH
    od = tmp % OD
    n = tmp // OD

    # Accumulator for OC values, BLOCK_W wide
    # shape [OC, BLOCK_W]
    oc_range = tl.arange(0, OC)
    # init with conv bias
    cb = tl.load(cb_ptr + oc_range)  # [OC]
    acc = cb[:, None] + tl.zeros([OC, BLOCK_W], dtype=tl.float32)

    in_spatial = ID * IH * IW
    in_nc_stride = IC * in_spatial

    # iterate over kernel
    for kd in tl.static_range(0, KD):
        id_num = od + PD - kd
        id_q = id_num // SD
        id_r = id_num - id_q * SD
        id_valid = (id_r == 0) & (id_q >= 0) & (id_q < ID)

        for kh in tl.static_range(0, KH):
            ih_num = oh + PH - kh
            ih_q = ih_num // SH
            ih_r = ih_num - ih_q * SH
            ih_valid = (ih_r == 0) & (ih_q >= 0) & (ih_q < IH)

            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PW - kw
                iw_q = iw_num // SW
                iw_r = iw_num - iw_q * SW
                iw_valid = (iw_r == 0) & (iw_q >= 0) & (iw_q < IW) & ow_mask

                spatial_valid = id_valid & ih_valid  # scalar
                full_mask = iw_valid & spatial_valid  # [BLOCK_W]

                # gather over IC
                for ic in range(0, IC):
                    # input offset: n*IC*ID*IH*IW + ic*ID*IH*IW + id_q*IH*IW + ih_q*IW + iw_q
                    in_off = (n * in_nc_stride + ic * in_spatial
                              + id_q * IH * IW + ih_q * IW + iw_q)
                    xv = tl.load(x_ptr + in_off, mask=full_mask, other=0.0)  # [BLOCK_W]

                    # weight offset: ic*OC*KD*KH*KW + oc*KD*KH*KW + kd*KH*KW + kh*KW + kw
                    w_off = (ic * OC * KD * KH * KW
                             + oc_range * KD * KH * KW
                             + kd * KH * KW + kh * KW + kw)
                    wv = tl.load(w_off + w_ptr)  # [OC]

                    acc += wv[:, None] * xv[None, :]

    # acc: [OC, BLOCK_W] - conv output (with bias added)
    # logsumexp over OC
    m = tl.max(acc, axis=0)                 # [BLOCK_W]
    e = tl.exp(acc - m[None, :])
    s = tl.sum(e, axis=0)
    lse = m + tl.log(s)                      # [BLOCK_W]

    # hardswish
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    # subtract scalar bias
    sb = tl.load(sb_ptr)
    y = hs - sb
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    # store output [N, 1, OD, OH, OW]
    out_off = (n * OD * OH * OW + od * OH * OW + oh * OW + ow_offs)
    tl.store(out_ptr + out_off, y, mask=ow_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        SD = SH = SW = self.stride
        PD = PH = PW = self.padding
        OC = self.out_channels

        OD = (ID - 1) * SD - 2 * PD + KD
        OH = (IH - 1) * SH - 2 * PH + KH
        OW = (IW - 1) * SW - 2 * PW + KW

        weight = self.conv_transpose.weight.contiguous()  # [IC, OC, KD, KH, KW]
        cb = self.conv_transpose.bias.contiguous() if self.conv_transpose.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)
        sb = self.bias.view(-1)[0:1].contiguous()

        out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_W = 32
        if OW <= 32:
            BLOCK_W = triton.next_power_of_2(OW)
            if BLOCK_W < 16:
                BLOCK_W = 16

        grid = (N * OD * OH, (OW + BLOCK_W - 1) // BLOCK_W)

        fused_convt_lse_kernel[grid](
            x, weight, cb, sb, out,
            N, IC, ID, IH, IW,
            OD, OH, OW,
            KD, KH, KW,
            SD, SH, SW,
            PD, PH, PW,
            OC=OC,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )
        return out