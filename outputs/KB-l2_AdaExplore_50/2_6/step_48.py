import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['N', 'C', 'D', 'H', 'W'],
)
@triton.jit
def fused_softmax_pool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    POOL: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    pid1 = pid // OW
    oh = pid1 % OH
    pid2 = pid1 // OH
    od = pid2 % OD
    n = pid2 // OD

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    WIN: tl.constexpr = POOL * POOL

    d_start = od * WIN
    h_start = oh * WIN
    w_start = ow * WIN

    max_vals = tl.full((BLOCK_C,), -float('inf'), dtype=tl.float32)

    # Input is in channels_last_3d layout: (N, D, H, W, C) contiguous
    DHWC = D * H * W * C
    HWC = H * W * C
    WC = W * C
    n_base = n * DHWC

    for dd in tl.static_range(0, WIN):
        for hh in tl.static_range(0, WIN):
            for ww in tl.static_range(0, WIN):
                d = d_start + dd
                h = h_start + hh
                w = w_start + ww
                base = n_base + d * HWC + h * WC + w * C
                ptrs = x_ptr + base + c_offs
                vals = tl.load(ptrs, mask=c_mask, other=-float('inf'))
                m = tl.max(vals, axis=0)
                vals_shift = vals - m
                e = tl.exp(vals_shift)
                e = tl.where(c_mask, e, 0.0)
                s = tl.sum(e, axis=0)
                sm = e / s
                sm = tl.where(c_mask, sm, -float('inf'))
                max_vals = tl.maximum(max_vals, sm)

    # Output in standard contiguous layout: (N, C, OD, OH, OW)
    out_base = n * C * OD * OH * OW + od * OH * OW + oh * OW + ow
    out_ptrs = out_ptr + out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptrs, max_vals, mask=c_mask)


def fused_softmax_pool(x, pool_kernel_size):
    N, C, D, H, W = x.shape
    POOL = pool_kernel_size
    WIN = POOL * POOL
    OD = D // WIN
    OH = H // WIN
    OW = W // WIN

    # Convert input to channels_last_3d so channel axis is unit-stride
    x_cl = x.to(memory_format=torch.channels_last_3d)

    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    # Choose BLOCK_C as next power of 2 >= C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * OD * OH * OW,)
    fused_softmax_pool_kernel[grid](
        x_cl, out,
        N, C, D, H, W,
        OD, OH, OW,
        POOL=POOL,
        BLOCK_C=BLOCK_C,
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
            return fused_softmax_pool(x, self.pool_kernel_size)
        else:
            x = torch.softmax(x, dim=1)
            x = F.max_pool3d(x, self.pool_kernel_size)
            x = F.max_pool3d(x, self.pool_kernel_size)
            return x