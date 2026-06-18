import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 8, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 8, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_OH': 4, 'BLOCK_OW': 16, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_OH': 8, 'BLOCK_OW': 16, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv2d_mish_mish_kernel(
    x_ptr,           # [N, IC, H, W] NCHW
    w_ptr,           # [OC, IC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC, OH, OW] NCHW
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    num_ow_tiles = tl.cdiv(OW, BLOCK_OW)
    pid_oh = pid_sp // num_ow_tiles
    pid_ow = pid_sp % num_ow_tiles

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)   # [BLOCK_OC]
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)   # [BLOCK_OH]
    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)   # [BLOCK_OW]

    oc_mask = offs_oc < OC
    oh_mask = offs_oh < OH
    ow_mask = offs_ow < OW

    # acc: [BLOCK_OC, BLOCK_OH, BLOCK_OW]
    acc = tl.zeros((BLOCK_OC, BLOCK_OH * BLOCK_OW), dtype=tl.float32)

    # Flatten spatial: spatial index in output tile
    offs_sp = offs_oh[:, None] * BLOCK_OW + offs_ow[None, :]  # [BLOCK_OH, BLOCK_OW]
    offs_sp_flat = tl.reshape(offs_sp, (BLOCK_OH * BLOCK_OW,))

    # Compute (oh, ow) for flat spatial offsets
    oh_flat = offs_sp_flat // BLOCK_OW
    ow_flat = offs_sp_flat % BLOCK_OW
    # absolute
    abs_oh = pid_oh * BLOCK_OH + oh_flat
    abs_ow = pid_ow * BLOCK_OW + ow_flat
    sp_mask = (abs_oh < OH) & (abs_ow < OW)

    x_batch_ptr = x_ptr + pid_b * IC * H * W

    offs_ic = tl.arange(0, BLOCK_IC)

    for ic_start in range(0, IC, BLOCK_IC):
        ic_idx = ic_start + offs_ic  # [BLOCK_IC]
        ic_mask = ic_idx < IC

        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = abs_oh + kh  # [SP]
                iw = abs_ow + kw  # [SP]
                # x: [BLOCK_IC, SP]
                x_offs = ic_idx[:, None] * (H * W) + ih[None, :] * W + iw[None, :]
                x_m = ic_mask[:, None] & sp_mask[None, :]
                x_tile = tl.load(x_batch_ptr + x_offs, mask=x_m, other=0.0)

                # w: [BLOCK_OC, BLOCK_IC]  layout [OC, IC, KH, KW]
                w_offs = offs_oc[:, None] * (IC * KH * KW) + ic_idx[None, :] * (KH * KW) + (kh * KW + kw)
                w_m = oc_mask[:, None] & ic_mask[None, :]
                w_tile = tl.load(w_ptr + w_offs, mask=w_m, other=0.0)

                acc += tl.dot(w_tile, x_tile)

    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]

    # double mish
    x1 = acc
    sp1 = tl.where(x1 > 20.0, x1, tl.log(1.0 + tl.exp(x1)))
    e1 = tl.exp(2.0 * sp1)
    t1 = (e1 - 1.0) / (e1 + 1.0)
    y = x1 * t1
    sp2 = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(y)))
    e2 = tl.exp(2.0 * sp2)
    t2 = (e2 - 1.0) / (e2 + 1.0)
    z = y * t2

    # Store: out [N, OC, OH, OW]
    out_batch_ptr = out_ptr + pid_b * OC * OH * OW
    out_offs = offs_oc[:, None] * (OH * OW) + abs_oh[None, :] * OW + abs_ow[None, :]
    out_m = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_batch_ptr + out_offs, z, mask=out_m)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        N, IC, H, W = x.shape
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1
        OC = self.out_channels

        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 64
        BLOCK_OH = 8
        BLOCK_OW = 32
        BLOCK_IC = 32

        num_ow_tiles = triton.cdiv(OW, BLOCK_OW)
        num_oh_tiles = triton.cdiv(OH, BLOCK_OH)
        grid = (N, triton.cdiv(OC, BLOCK_OC), num_oh_tiles * num_ow_tiles)

        conv2d_mish_mish_kernel[grid](
            x, w, b, out,
            N, IC, H, W, OC, OH, OW,
            KH=KH, KW=KW,
            BLOCK_OC=BLOCK_OC, BLOCK_OH=BLOCK_OH, BLOCK_OW=BLOCK_OW,
            BLOCK_IC=BLOCK_IC,
            num_warps=4, num_stages=3,
        )
        return out