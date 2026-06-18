import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 62}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OW': 62}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_OW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OW': 32}, num_warps=4, num_stages=3),
    ],
    key=['IC', 'OD', 'OH', 'OW', 'OC'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr,        # [N, IC, ID, IH, IW]
    w_ptr,        # [OC, IC*KD*KH*KW]
    b_ptr,        # [OC]
    sum_ptr,      # [OC]
    out_ptr,      # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,         # IC*KD*KH*KW
    BLOCK_OC: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_n = tl.program_id(0)              # batch
    pid_oc = tl.program_id(1)             # OC tile
    pid_sp = tl.program_id(2)             # spatial tile (od, oh, ow_tile)

    OW_TILES = (OW + BLOCK_OW - 1) // BLOCK_OW
    od = pid_sp // (OH * OW_TILES)
    rem = pid_sp % (OH * OW_TILES)
    oh = rem // OW_TILES
    ow_tile = rem % OW_TILES

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)   # [BLOCK_OC]
    offs_ow = ow_tile * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]

    oc_mask = offs_oc < OC
    ow_mask = offs_ow < OW

    KDHW: tl.constexpr = KD * KH * KW

    acc = tl.zeros((BLOCK_OC, BLOCK_OW), dtype=tl.float32)

    x_batch_off = pid_n * IC * ID * IH * IW
    ih_base = oh
    id_base = od

    for ic in range(0, IC):
        x_ic_off = x_batch_off + ic * (ID * IH * IW)
        w_ic_off = ic * KDHW
        for kk in tl.static_range(0, KDHW):
            kd = kk // (KH * KW)
            rkw = kk % (KH * KW)
            kh = rkw // KW
            kw = rkw % KW

            id_ = id_base + kd
            ih_ = ih_base + kh
            iw_ = offs_ow + kw  # [BLOCK_OW]

            in_off = x_ic_off + id_ * (IH * IW) + ih_ * IW + iw_
            x_val = tl.load(x_ptr + in_off, mask=ow_mask, other=0.0)  # [BLOCK_OW]

            w_col = tl.load(w_ptr + offs_oc * K + (w_ic_off + kk),
                            mask=oc_mask, other=0.0)  # [BLOCK_OC]

            acc += w_col[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    acc = tl.where(acc > 0, acc, acc * 0.2)

    s_val = tl.load(sum_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = acc + s_val[:, None]

    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)

    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    ODHW = OD * OH * OW
    out_base = (pid_n * OC * ODHW
                + offs_oc[:, None] * ODHW
                + od * (OH * OW)
                + oh * OW
                + offs_ow[None, :])
    out_mask = oc_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptr + out_base, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        K = IC * KD * KH * KW

        # pack weight to [OC, K]
        w = self.conv.weight.contiguous().view(OC, K).contiguous()
        b = self.conv.bias.contiguous()
        s = self.sum_tensor.contiguous().view(-1).contiguous()

        out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        def grid(meta):
            ow_tiles = triton.cdiv(OW, meta['BLOCK_OW'])
            oc_tiles = triton.cdiv(OC, meta['BLOCK_OC'])
            return (N, oc_tiles, OD * OH * ow_tiles)

        conv3d_fused_kernel[grid](
            x, w, b, s, out,
            N, IC, ID, IH, IW,
            OD, OH, OW,
            OC=OC,
            KD=KD, KH=KH, KW=KW,
            K=K,
        )
        return out