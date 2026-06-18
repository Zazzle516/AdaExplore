import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_pool_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    POOL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, od, oh, ow)
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    # base indices in input
    d0 = od * POOL
    h0 = oh * POOL
    w0 = ow * POOL

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # accumulator for max-pool over POOL^3 window of softmax values
    max_val = tl.zeros([BLOCK_C], dtype=tl.float32) - float('inf')

    # iterate the pool window
    for dd in tl.static_range(0, POOL):
        for hh in tl.static_range(0, POOL):
            for ww in tl.static_range(0, POOL):
                d = d0 + dd
                h = h0 + hh
                w = w0 + ww
                # base offset for this (n, *, d, h, w)
                base = ((n * C) * D + d) * H * W + h * W + w
                # stride to next channel = D*H*W
                ptrs = in_ptr + base + c_offs * (D * H * W)
                vals = tl.load(ptrs, mask=c_mask, other=-float('inf'))
                # compute softmax along C
                m = tl.max(vals, axis=0)
                ex = tl.exp(vals - m)
                ex = tl.where(c_mask, ex, 0.0)
                s = tl.sum(ex, axis=0)
                sm = ex / s
                sm = tl.where(c_mask, sm, -float('inf'))
                max_val = tl.maximum(max_val, sm)

    # store
    out_base = ((n * C) * OD + od) * OH * OW + oh * OW + ow
    out_ptrs = out_ptr + out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptrs, max_val, mask=c_mask)


def fused_softmax_pool(x, pool_total):
    N, C, D, H, W = x.shape
    OD = D // pool_total
    OH = H // pool_total
    OW = W // pool_total
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    grid = (N * OD * OH * OW,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        POOL=pool_total,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_total = pool_kernel_size * pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        x = x.contiguous()
        # fused softmax + double maxpool (combined as a pool of stride pool*pool)
        out = fused_softmax_pool(x, self.pool_total)
        return out