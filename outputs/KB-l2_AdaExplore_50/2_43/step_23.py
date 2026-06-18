import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


torch.backends.cudnn.benchmark = True


@triton.jit
def fused_pool_lse_relu_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_C: tl.constexpr,
    TILE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid indexes (n, do, ho, wo_tile)
    Wo_tiles = (Wo + TILE_W - 1) // TILE_W
    wo_tile = pid % Wo_tiles
    tmp = pid // Wo_tiles
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    n = tmp // Do

    wo_start = wo_tile * TILE_W
    d0 = do * 2
    h0 = ho * 2

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    w_offs = tl.arange(0, TILE_W)
    wo_idx = wo_start + w_offs
    w_mask = wo_idx < Wo

    base = n * C * D * H * W
    DHW = D * H * W
    HW = H * W

    neg_inf = float('-inf')
    # max_vals shape: [BLOCK_C, TILE_W]
    max_vals = tl.full([BLOCK_C, TILE_W], neg_inf, dtype=tl.float32)

    for dd in tl.static_range(2):
        for hh in tl.static_range(2):
            for ww in tl.static_range(2):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = wo_idx * 2 + ww
                offs = (base
                        + c_offs[:, None] * DHW
                        + d_idx * HW
                        + h_idx * W
                        + w_idx[None, :])
                m_load = c_mask[:, None] & w_mask[None, :]
                v = tl.load(in_ptr + offs, mask=m_load, other=neg_inf)
                max_vals = tl.maximum(max_vals, v)

    # reduce across C
    masked = tl.where(c_mask[:, None], max_vals, neg_inf)
    m = tl.max(masked, axis=0)  # [TILE_W]
    exps = tl.where(c_mask[:, None], tl.exp(max_vals - m[None, :]), 0.0)
    s = tl.sum(exps, axis=0)
    lse = m + tl.log(s)
    out_val = tl.maximum(lse, 0.0)

    out_offset = n * (Do * Ho * Wo) + do * (Ho * Wo) + ho * Wo + wo_idx
    tl.store(out_ptr + out_offset, out_val, mask=w_mask)


def fused_pool_lse_relu(x):
    N, C, D, H, W = x.shape
    Do = D // 2
    Ho = H // 2
    Wo = W // 2
    x = x.contiguous()
    out = torch.empty((N, 1, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    TILE_W = 16
    Wo_tiles = (Wo + TILE_W - 1) // TILE_W

    grid = (N * Do * Ho * Wo_tiles,)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Do, Ho, Wo,
        BLOCK_C=BLOCK_C,
        TILE_W=TILE_W,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_pool_lse_relu(x)
        return x