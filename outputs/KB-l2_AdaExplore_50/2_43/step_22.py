import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_pool_lse_relu_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid indexes (n, do, ho, wo)
    wo = pid % Wo
    tmp = pid // Wo
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    n = tmp // Do

    d0 = do * 2
    h0 = ho * 2
    w0 = wo * 2

    # For each channel, compute max over 2x2x2 pool window
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # base offset for (n, c=0, d0, h0, w0)
    base = n * C * D * H * W

    # accumulate max over 8 positions
    neg_inf = float('-inf')
    max_vals = tl.full([BLOCK_C], neg_inf, dtype=tl.float32)

    for dd in tl.static_range(2):
        for hh in tl.static_range(2):
            for ww in tl.static_range(2):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                offs = base + c_offs * (D * H * W) + d_idx * (H * W) + h_idx * W + w_idx
                v = tl.load(in_ptr + offs, mask=c_mask, other=neg_inf)
                max_vals = tl.maximum(max_vals, v)

    # logsumexp over channels
    # numerically stable
    m = tl.max(tl.where(c_mask, max_vals, neg_inf), axis=0)
    exps = tl.where(c_mask, tl.exp(max_vals - m), 0.0)
    s = tl.sum(exps, axis=0)
    lse = m + tl.log(s)
    # relu
    out_val = tl.maximum(lse, 0.0)

    out_offset = n * (Do * Ho * Wo) + do * (Ho * Wo) + ho * Wo + wo
    tl.store(out_ptr + out_offset, out_val)


def fused_pool_lse_relu(x):
    # x: (N, C, D, H, W) - already convolved
    N, C, D, H, W = x.shape
    Do = D // 2
    Ho = H // 2
    Wo = W // 2
    x = x.contiguous()
    out = torch.empty((N, 1, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    # pick BLOCK_C as next power of 2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * Do * Ho * Wo,)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Do, Ho, Wo,
        BLOCK_C=BLOCK_C,
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