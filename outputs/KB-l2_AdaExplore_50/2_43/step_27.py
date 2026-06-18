import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'TILE_W': 8}, num_warps=2, num_stages=2),
        triton.Config({'TILE_W': 8}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 16}, num_warps=2, num_stages=2),
        triton.Config({'TILE_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 16}, num_warps=8, num_stages=2),
        triton.Config({'TILE_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'TILE_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'TILE_W': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'Do', 'Ho', 'Wo'],
)
@triton.jit
def fused_pool_lse_relu_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_C: tl.constexpr,
    TILE_W: tl.constexpr,
):
    # grid: (ceil(Wo/TILE_W), Ho*Do, N)
    pid_w = tl.program_id(0)
    pid_hd = tl.program_id(1)
    pid_n = tl.program_id(2)

    ho = pid_hd % Ho
    do = pid_hd // Ho
    n = pid_n

    w_offs = pid_w * TILE_W + tl.arange(0, TILE_W)
    w_mask = w_offs < Wo

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    neg_inf = float('-inf')
    DHW = D * H * W
    HW = H * W

    d0 = do * 2
    h0 = ho * 2
    w0 = w_offs * 2  # [TILE_W]

    # base offset per (n, c=0, d0, h0, w0)
    # shape [BLOCK_C, TILE_W]
    n_base = n * C * DHW
    c_base = c_offs[:, None] * DHW  # [BLOCK_C, 1]
    sp_base = d0 * HW + h0 * W + w0[None, :]  # [1, TILE_W]
    base = n_base + c_base + sp_base

    mask2d = c_mask[:, None] & w_mask[None, :]

    v0 = tl.load(in_ptr + base, mask=mask2d, other=neg_inf)
    v1 = tl.load(in_ptr + base + 1, mask=mask2d, other=neg_inf)
    v2 = tl.load(in_ptr + base + W, mask=mask2d, other=neg_inf)
    v3 = tl.load(in_ptr + base + W + 1, mask=mask2d, other=neg_inf)
    v4 = tl.load(in_ptr + base + HW, mask=mask2d, other=neg_inf)
    v5 = tl.load(in_ptr + base + HW + 1, mask=mask2d, other=neg_inf)
    v6 = tl.load(in_ptr + base + HW + W, mask=mask2d, other=neg_inf)
    v7 = tl.load(in_ptr + base + HW + W + 1, mask=mask2d, other=neg_inf)

    max_vals = tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)),
                          tl.maximum(tl.maximum(v4, v5), tl.maximum(v6, v7)))
    # max_vals: [BLOCK_C, TILE_W]

    masked = tl.where(mask2d, max_vals, neg_inf)
    m = tl.max(masked, axis=0)  # [TILE_W]
    exps = tl.where(mask2d, tl.exp(max_vals - m[None, :]), 0.0)
    s = tl.sum(exps, axis=0)  # [TILE_W]
    lse = m + tl.log(s)
    out_val = tl.maximum(lse, 0.0)

    out_idx = ((n * Do + do) * Ho + ho) * Wo + w_offs
    tl.store(out_ptr + out_idx, out_val, mask=w_mask)


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

    grid = lambda meta: ((Wo + meta['TILE_W'] - 1) // meta['TILE_W'], Ho * Do, N)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Do, Ho, Wo,
        BLOCK_C=BLOCK_C,
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