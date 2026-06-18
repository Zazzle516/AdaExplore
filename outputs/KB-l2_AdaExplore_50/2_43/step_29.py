import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'TILE': 4}, num_warps=2, num_stages=2),
        triton.Config({'TILE': 4}, num_warps=4, num_stages=2),
        triton.Config({'TILE': 8}, num_warps=2, num_stages=2),
        triton.Config({'TILE': 8}, num_warps=4, num_stages=2),
        triton.Config({'TILE': 16}, num_warps=4, num_stages=2),
        triton.Config({'TILE': 16}, num_warps=8, num_stages=2),
        triton.Config({'TILE': 32}, num_warps=4, num_stages=2),
        triton.Config({'TILE': 32}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'Do', 'Ho', 'Wo'],
)
@triton.jit
def fused_pool_lse_relu_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    TOTAL,
    BLOCK_C: tl.constexpr,
    TILE: tl.constexpr,
):
    pid = tl.program_id(0)
    base_idx = pid * TILE

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    neg_inf = float('-inf')
    DHW = D * H * W
    HW = H * W

    for t in tl.static_range(TILE):
        idx = base_idx + t
        if idx < TOTAL:
            wo = idx % Wo
            tmp = idx // Wo
            ho = tmp % Ho
            tmp = tmp // Ho
            do = tmp % Do
            n = tmp // Do

            d0 = do * 2
            h0 = ho * 2
            w0 = wo * 2

            base = n * C * DHW + d0 * HW + h0 * W + w0
            c_base = c_offs * DHW

            v0 = tl.load(in_ptr + base + c_base, mask=c_mask, other=neg_inf)
            v1 = tl.load(in_ptr + base + c_base + 1, mask=c_mask, other=neg_inf)
            v2 = tl.load(in_ptr + base + c_base + W, mask=c_mask, other=neg_inf)
            v3 = tl.load(in_ptr + base + c_base + W + 1, mask=c_mask, other=neg_inf)
            v4 = tl.load(in_ptr + base + c_base + HW, mask=c_mask, other=neg_inf)
            v5 = tl.load(in_ptr + base + c_base + HW + 1, mask=c_mask, other=neg_inf)
            v6 = tl.load(in_ptr + base + c_base + HW + W, mask=c_mask, other=neg_inf)
            v7 = tl.load(in_ptr + base + c_base + HW + W + 1, mask=c_mask, other=neg_inf)

            max_vals = tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)),
                                  tl.maximum(tl.maximum(v4, v5), tl.maximum(v6, v7)))

            m = tl.max(tl.where(c_mask, max_vals, neg_inf), axis=0)
            exps = tl.where(c_mask, tl.exp(max_vals - m), 0.0)
            s = tl.sum(exps, axis=0)
            lse = m + tl.log(s)
            out_val = tl.maximum(lse, 0.0)

            tl.store(out_ptr + idx, out_val)


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

    total = N * Do * Ho * Wo
    grid = lambda meta: ((total + meta['TILE'] - 1) // meta['TILE'],)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Do, Ho, Wo,
        total,
        BLOCK_C=BLOCK_C,
    )
    return out


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_pool_lse_relu(x)
        return x