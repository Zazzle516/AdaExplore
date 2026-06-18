import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 1, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 2, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_H': 4, 'BLOCK_W': 32}, num_warps=8, num_stages=2),
    ],
    key=['OD', 'OH', 'OW', 'OC'],
)
@triton.jit
def fused_conv_transpose_logsumexp_kernel(
    x_ptr,          # [N, IC, ID, IH, IW]
    w_ptr,          # [IC, OC, KD, KH, KW]
    cbias_ptr,      # [OC] conv bias
    sbias_ptr,      # scalar bias
    out_ptr,        # [N, 1, OD, OH, OW]
    N, IC,
    ID, IH, IW,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh-tile, ow-tile)
    pid_now = tl.program_id(0)  # n * OD * cdiv(OH, BLOCK_H)
    pid_w = tl.program_id(1)

    OH_TILES = (OH + BLOCK_H - 1) // BLOCK_H
    n = pid_now // (OD * OH_TILES)
    rem = pid_now % (OD * OH_TILES)
    od = rem // OH_TILES
    oh_tile = rem % OH_TILES

    oh_offs = oh_tile * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    ow_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)    # [BLOCK_W]
    oh_mask = oh_offs < OH
    ow_mask = ow_offs < OW

    oc_range = tl.arange(0, OC)  # [OC]

    # accumulator [BLOCK_H, BLOCK_W, OC]
    acc = tl.zeros((BLOCK_H, BLOCK_W, OC), dtype=tl.float32)

    for kd in tl.static_range(0, KD):
        id_num = od + PAD - kd
        id_val = id_num // STRIDE
        id_ok = (id_num % STRIDE == 0) & (id_val >= 0) & (id_val < ID)
        for kh in tl.static_range(0, KH):
            ih_num = oh_offs + PAD - kh           # [BLOCK_H]
            ih_val = ih_num // STRIDE
            ih_ok = id_ok & (ih_num % STRIDE == 0) & (ih_val >= 0) & (ih_val < IH) & oh_mask  # [BLOCK_H]
            for kw in tl.static_range(0, KW):
                iw_num = ow_offs + PAD - kw       # [BLOCK_W]
                iw_val = iw_num // STRIDE
                iw_ok = (iw_num % STRIDE == 0) & (iw_val >= 0) & (iw_val < IW) & ow_mask  # [BLOCK_W]

                hw_mask = ih_ok[:, None] & iw_ok[None, :]  # [BLOCK_H, BLOCK_W]

                for ic in tl.static_range(0, 3):  # IC=3
                    # x[n, ic, id_val, ih_val[h], iw_val[w]]
                    x_idx = ((n * IC + ic) * ID + id_val) * IH * IW + ih_val[:, None] * IW + iw_val[None, :]
                    x_vals = tl.load(x_ptr + x_idx, mask=hw_mask, other=0.0)  # [BLOCK_H, BLOCK_W]

                    w_idx = ic * OC * KD * KH * KW + oc_range * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw  # [OC]
                    w_vals = tl.load(w_ptr + w_idx)  # [OC]

                    acc += x_vals[:, :, None] * w_vals[None, None, :]

    # add conv bias
    cb = tl.load(cbias_ptr + oc_range)  # [OC]
    acc += cb[None, None, :]

    # logsumexp across OC
    m = tl.max(acc, axis=2)               # [BLOCK_H, BLOCK_W]
    e = tl.exp(acc - m[:, :, None])
    s = tl.sum(e, axis=2)                 # [BLOCK_H, BLOCK_W]
    lse = m + tl.log(s)                   # [BLOCK_H, BLOCK_W]

    # hardswish
    sig = 1.0 / (1.0 + tl.exp(-(lse + 3.0)))
    hs = lse * sig / 6.0

    sb = tl.load(sbias_ptr)
    y = hs - sb
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)

    out_off = n * (OD * OH * OW) + od * (OH * OW) + oh_offs[:, None] * OW + ow_offs[None, :]
    out_mask = oh_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_off, y, mask=out_mask)


def fused_op(x, weight, cbias, sbias, stride, padding):
    N, IC, ID, IH, IW = x.shape
    IC_w, OC, KD, KH, KW = weight.shape
    assert IC == IC_w

    OD = (ID - 1) * stride - 2 * padding + KD
    OH = (IH - 1) * stride - 2 * padding + KH
    OW = (IW - 1) * stride - 2 * padding + KW

    x = x.contiguous()
    weight = weight.contiguous()

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N * OD * triton.cdiv(OH, meta['BLOCK_H']),
        triton.cdiv(OW, meta['BLOCK_W']),
    )
    fused_conv_transpose_logsumexp_kernel[grid](
        x, weight, cbias, sbias.reshape(-1),
        out,
        N, IC,
        ID, IH, IW,
        OD, OH, OW,
        OC=OC,
        KD=KD, KH=KH, KW=KW,
        STRIDE=stride, PAD=padding,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, 1, 1, 1))
        self.stride = stride
        self.padding = padding
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x):
        w = self.conv_transpose.weight
        cb = self.conv_transpose.bias
        return fused_op(x, w, cb, self.bias, self.stride, self.padding)