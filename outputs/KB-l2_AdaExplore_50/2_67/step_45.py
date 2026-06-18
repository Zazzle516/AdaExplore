import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    KSIZE: tl.constexpr,
    KSIZE_P2: tl.constexpr,
    BLOCK_SPATIAL: tl.constexpr,
    NUM_TILES: tl.constexpr,
):
    # grid: (N, OC, NUM_TILES)
    n = tl.program_id(0)
    oc = tl.program_id(1)
    tile = tl.program_id(2)

    OHOW: tl.constexpr = OH * OW
    inv = 1.0 / OHOW

    # Load weights for this oc
    k_offs = tl.arange(0, KSIZE_P2)
    k_mask = k_offs < KSIZE
    w_base = oc * IC_C * KH * KW
    w_vals = tl.load(w_ptr + w_base + k_offs, mask=k_mask, other=0.0)

    bias = tl.load(b_ptr + oc)

    # decode k_offs into (ic, kh, kw)
    k_safe = tl.where(k_mask, k_offs, 0)
    ic_idx = k_safe // (KH * KW)
    rem = k_safe % (KH * KW)
    kh_idx = rem // KW
    kw_idx = rem % KW

    # This tile processes spatial positions in [tile*TILE_SIZE, (tile+1)*TILE_SIZE)
    TILE_SIZE: tl.constexpr = (OHOW + NUM_TILES - 1) // NUM_TILES
    BLOCKS_PER_TILE: tl.constexpr = (TILE_SIZE + BLOCK_SPATIAL - 1) // BLOCK_SPATIAL

    tile_start = tile * TILE_SIZE
    acc = tl.zeros((), dtype=tl.float32)

    for blk in range(0, BLOCKS_PER_TILE):
        sp_offs = tile_start + blk * BLOCK_SPATIAL + tl.arange(0, BLOCK_SPATIAL)
        sp_mask = sp_offs < OHOW
        oh = sp_offs // OW
        ow = sp_offs % OW

        ih = oh[:, None] + kh_idx[None, :]
        iw = ow[:, None] + kw_idx[None, :]
        ic_b = ic_idx[None, :] + tl.zeros((BLOCK_SPATIAL, KSIZE_P2), dtype=tl.int32)

        x_idx = ((n * IC + ic_b) * IH + ih) * IW + iw
        x_mask = sp_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        prod = x_vals * w_vals[None, :]
        conv_out = tl.sum(prod, axis=1) + bias

        # GELU (tanh approximation)
        c0 = 0.7978845608028654
        c1 = 0.044715
        x3 = conv_out * conv_out * conv_out
        inner = c0 * (conv_out + c1 * x3)
        t = 2.0 * tl.sigmoid(2.0 * inner) - 1.0
        gelu = 0.5 * conv_out * (1.0 + t)

        gelu = tl.where(sp_mask, gelu, 0.0)
        acc += tl.sum(gelu, axis=0)

    # atomic add the partial sum * inv into output
    partial = acc * inv
    tl.atomic_add(out_ptr + n * OC + oc, partial)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1

        out = torch.zeros((N, OC), device=x.device, dtype=x.dtype)

        BLOCK_SPATIAL = 512
        NUM_TILES = 8
        KSIZE = IC * KH * KW
        KSIZE_P2 = 1
        while KSIZE_P2 < KSIZE:
            KSIZE_P2 *= 2
        grid = (N, OC, NUM_TILES)
        conv_gelu_avgpool_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC,
            OH, OW,
            KH, KW,
            IC,
            KSIZE,
            KSIZE_P2,
            BLOCK_SPATIAL,
            NUM_TILES,
            num_warps=4,
            num_stages=2,
        )
        return out