import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_sum_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh, ow_tile)
    pid = tl.program_id(0)
    n_ow_tiles = (OW + BLOCK_W - 1) // BLOCK_W
    ow_tile = pid % n_ow_tiles
    tmp = pid // n_ow_tiles
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    ow_offs = ow_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    ow_mask = ow_offs < OW

    d_base = od * 6
    h_base = oh * 6
    w_base = ow_offs * 6  # shape [BLOCK_W]

    c_offs = tl.arange(0, BLOCK_C)

    # accumulator per ow
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_idx = c_start + c_offs
        c_mask = c_idx < C

        # ch_max: [BLOCK_C, BLOCK_W]
        ch_max = tl.full([BLOCK_C, BLOCK_W], -float('inf'), dtype=tl.float32)
        nc_base = (n * C + c_idx) * D  # [BLOCK_C]
        for dd in range(6):
            d_in = d_base + dd
            d_ok = d_in < D
            dh_base = (nc_base + d_in) * H  # [BLOCK_C]
            for hh in range(6):
                h_in = h_base + hh
                h_ok = h_in < H
                row_base = (dh_base + h_in) * W  # [BLOCK_C]
                for ww in range(6):
                    w_in = w_base + ww  # [BLOCK_W]
                    w_ok = w_in < W
                    offset = row_base[:, None] + w_in[None, :]  # [BLOCK_C, BLOCK_W]
                    mask = c_mask[:, None] & ow_mask[None, :] & (d_ok & h_ok) & w_ok[None, :]
                    val = tl.load(in_ptr + offset, mask=mask, other=-float('inf'))
                    ch_max = tl.maximum(ch_max, val)
        ch_max = tl.where(c_mask[:, None], ch_max, 0.0)
        acc += tl.sum(ch_max, axis=0)

    out_off = ((n * OD + od) * OH + oh) * OW + ow_offs
    tl.store(out_ptr + out_off, acc, mask=ow_mask)


def fused_pool_sum(x):
    # x: (N, C, D, H, W) -- output of conv_transpose
    # apply maxpool(2) then maxpool(3) then sum over channel
    N, C, D, H, W = x.shape
    # maxpool(2) with no padding: floor(D/2)
    D1 = D // 2
    H1 = H // 2
    W1 = W // 2
    # maxpool(3) no padding: floor(D1/3)
    OD = D1 // 3
    OH = H1 // 3
    OW = W1 // 3

    # Effective: for each output element at (od,oh,ow), we need max over
    # input region [od*6 : od*6+6, ...] but only over the part that maxpool1 actually covers
    # maxpool1 covers d in [0, D1*2). So we need d_in < D1*2 (and < D).
    # maxpool2 covers d1 in [0, OD*3). So d1 = od*3 + i for i in [0,3], d = d1*2 + j for j in [0,2]
    # => d = od*6 + i*2 + j  for i in [0,3), j in [0,2). All such d are in [od*6, od*6+6).
    # So the 6x6x6 region is exact provided d < D.

    D_lim = min(OD * 6, D)
    H_lim = min(OH * 6, H)
    W_lim = min(OW * 6, W)

    x = x.contiguous()
    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = 64
    BLOCK_W = 4
    n_ow_tiles = (OW + BLOCK_W - 1) // BLOCK_W
    grid = (N * OD * OH * n_ow_tiles,)

    fused_pool_sum_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        BLOCK_C=BLOCK_C,
        BLOCK_W=BLOCK_W,
        num_warps=4,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_pool_sum(x)
        return x