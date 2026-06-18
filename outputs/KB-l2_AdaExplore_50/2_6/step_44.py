import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_pool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    POOL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # program ids: one program per (n, od, oh, ow)
    pid = tl.program_id(0)
    ow = pid % OW
    pid1 = pid // OW
    oh = pid1 % OH
    pid2 = pid1 // OH
    od = pid2 % OD
    n = pid2 // OD

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    POOL_VOL: tl.constexpr = POOL * POOL * POOL * POOL * POOL * POOL  # not used

    # We need to compute, for each channel c, the max over the (POOL^2)^... actually
    # two consecutive maxpool with kernel POOL and stride POOL is equivalent to a
    # single maxpool with kernel POOL*POOL and stride POOL*POOL.
    # Window size in input space (after conv) = POOL*POOL on each spatial dim.
    # Window start = (od, oh, ow) * (POOL*POOL).
    WIN: tl.constexpr = POOL * POOL

    d_start = od * WIN
    h_start = oh * WIN
    w_start = ow * WIN

    # Accumulator: per-channel max value
    max_vals = tl.full((BLOCK_C,), -float('inf'), dtype=tl.float32)

    # First pass: find max across channels at each spatial position to be able to
    # compute softmax stably. But softmax is per spatial position (over channels).
    # Then we max-pool the softmax output across the spatial window per channel.
    # So for each spatial (d,h,w) in the window, we:
    #   - load x[n, :, d, h, w]
    #   - compute softmax over channels -> sm[c]
    #   - update max_vals[c] = max(max_vals[c], sm[c])

    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                d = d_start + dd
                h = h_start + hh
                w = w_start + ww
                in_bounds = (d < D) & (h < H) & (w < W)
                # Load channel vector
                base = ((n * C + 0) * D + d) * H * W + h * W + w
                ptrs = x_ptr + base + c_offs * (D * H * W)
                vals = tl.load(ptrs, mask=c_mask & in_bounds, other=-float('inf'))
                # Softmax over channels
                m = tl.max(vals, axis=0)
                vals_shift = vals - m
                e = tl.exp(vals_shift)
                e = tl.where(c_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                # mask invalid positions to -inf so they don't affect max
                sm = tl.where(c_mask & in_bounds, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

    # Store output: out[n, c, od, oh, ow]
    out_base = ((n * C + 0) * OD + od) * OH * OW + oh * OW + ow
    out_ptrs = out_ptr + out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_softmax_pool(x, pool_kernel_size):
    N, C, D, H, W = x.shape
    POOL = pool_kernel_size
    WIN = POOL * POOL
    OD = D // WIN
    OH = H // WIN
    OW = W // WIN

    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    # Choose BLOCK_C as next power of 2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * OD * OH * OW,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        POOL=POOL,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv(x)
        # Fused softmax(dim=1) + two MaxPool3d(pool_kernel_size)
        # Need shape divisible by pool*pool on each spatial dim
        WIN = self.pool_kernel_size * self.pool_kernel_size
        N, C, D, H, W = x.shape
        if D % WIN == 0 and H % WIN == 0 and W % WIN == 0:
            return fused_softmax_pool(x.contiguous(), self.pool_kernel_size)
        else:
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x