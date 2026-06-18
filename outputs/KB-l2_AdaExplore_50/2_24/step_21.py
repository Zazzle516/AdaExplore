import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC,
    D: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr, OC_PAD: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    TILE_H: tl.constexpr, TILE_W: tl.constexpr,
):
    # grid: (N, ceil(OH/TILE_H) * ceil(OW/TILE_W))
    n = tl.program_id(0)
    tile_id = tl.program_id(1)

    nW_tiles = (OW + TILE_W - 1) // TILE_W
    th = tile_id // nW_tiles
    tw = tile_id % nW_tiles

    oh_base = th * TILE_H
    ow_base = tw * TILE_W

    offs_h = oh_base + tl.arange(0, TILE_H)  # (TILE_H,)
    offs_w = ow_base + tl.arange(0, TILE_W)  # (TILE_W,)
    mask_h = offs_h < OH
    mask_w = offs_w < OW

    # Build a (TILE_H, TILE_W) spatial mask
    mask_hw = mask_h[:, None] & mask_w[None, :]
    # flatten to (TILE_H*TILE_W,)
    THW = TILE_H * TILE_W

    offs_oc = tl.arange(0, OC_PAD)
    mask_oc = offs_oc < OC

    # bias broadcast
    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)  # (OC_PAD,)

    INF = float('inf')
    # min_val shape (OC_PAD, TILE_H, TILE_W)
    min_val = tl.full((OC_PAD, TILE_H, TILE_W), INF, dtype=tl.float32)

    for od in range(0, OD):
        acc = tl.zeros((OC_PAD, TILE_H, TILE_W), dtype=tl.float32)
        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = offs_h + kh  # (TILE_H,)
                    for kw in range(0, KW):
                        iw = offs_w + kw  # (TILE_W,)
                        # x offsets shape (TILE_H, TILE_W)
                        x_off = ((n * IC + ic) * D + id_) * H * W + ih[:, None] * W + iw[None, :]
                        x_val = tl.load(x_ptr + x_off, mask=mask_hw, other=0.0)  # (TILE_H, TILE_W)
                        # weight offsets shape (OC_PAD,)
                        w_off = ((offs_oc * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # (OC_PAD,)
                        # outer product accumulate
                        acc += w_val[:, None, None] * x_val[None, :, :]
        acc = acc + bias[:, None, None]
        min_val = tl.minimum(min_val, acc)

    # Apply mask: set invalid OC to -inf so they don't affect max/sum
    NEG_INF = float('-inf')
    min_val = tl.where(mask_oc[:, None, None], min_val, NEG_INF)

    # softmax along OC axis (axis 0)
    m = tl.max(min_val, axis=0)  # (TILE_H, TILE_W)
    e = tl.exp(min_val - m[None, :, :])
    e = tl.where(mask_oc[:, None, None], e, 0.0)
    s = tl.sum(e, axis=0)  # (TILE_H, TILE_W)
    out_vals = e / s[None, :, :]

    # store: out shape (N, OC, OH, OW)
    out_off = ((n * OC + offs_oc[:, None, None]) * OH + offs_h[None, :, None]) * OW + offs_w[None, None, :]
    store_mask = mask_oc[:, None, None] & mask_h[None, :, None] & mask_w[None, None, :]
    tl.store(out_ptr + out_off, out_vals, mask=store_mask)


def conv3d_min_softmax(x, weight, bias):
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = D - KD + 1
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

    # pad OC to next power of 2
    OC_PAD = 1
    while OC_PAD < OC:
        OC_PAD *= 2

    TILE_H = 8
    TILE_W = 8
    nH = (OH + TILE_H - 1) // TILE_H
    nW = (OW + TILE_W - 1) // TILE_W

    grid = (N, nH * nW)

    conv3d_min_softmax_kernel[grid](
        x, weight, bias, out,
        N, IC,
        D, H, W,
        OC, OC_PAD,
        OD, OH, OW,
        KD, KH, KW,
        TILE_H, TILE_W,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()
        if self.dim == 2:
            return conv3d_min_softmax(x, w, b)
        else:
            x = self.conv(x)
            x = torch.min(x, dim=self.dim)[0]
            x = torch.softmax(x, dim=1)
            return x