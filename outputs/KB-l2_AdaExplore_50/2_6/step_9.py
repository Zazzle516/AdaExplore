import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W', 'POOL'],
)
@triton.jit
def fused_softmax_pool_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    POOL: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    # one program per (n, od, oh, ow_block)
    pid = tl.program_id(0)
    num_ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    owb = pid % num_ow_blocks
    tmp = pid // num_ow_blocks
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    ow_offs = owb * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    # base indices in input
    d0 = od * POOL
    h0 = oh * POOL

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # hoist constants
    DHW = D * H * W
    HW = H * W
    n_base = n * C * DHW

    # 2D accumulator [BLOCK_C, BLOCK_OW]
    max_val = tl.zeros([BLOCK_C, BLOCK_OW], dtype=tl.float32) - float('inf')

    mask_2d = c_mask[:, None] & ow_mask[None, :]

    # iterate the pool window
    for dd in tl.static_range(0, POOL):
        d = d0 + dd
        d_base = n_base + d * HW
        for hh in tl.static_range(0, POOL):
            h = h0 + hh
            h_base = d_base + h * W
            for ww in tl.static_range(0, POOL):
                # w positions: ow_offs * POOL + ww
                w_pos = ow_offs * POOL + ww
                # ptrs[c, k] = in_ptr + h_base + w_pos[k] + c_offs[c] * DHW
                ptrs = in_ptr + h_base + w_pos[None, :] + c_offs[:, None] * DHW
                vals = tl.load(ptrs, mask=mask_2d, other=-float('inf'))
                # softmax along C (axis=0)
                m = tl.max(vals, axis=0)  # [BLOCK_OW]
                ex = tl.exp(vals - m[None, :])
                ex = tl.where(mask_2d, ex, 0.0)
                s = tl.sum(ex, axis=0)  # [BLOCK_OW]
                sm = ex / s[None, :]
                sm = tl.where(mask_2d, sm, -float('inf'))
                max_val = tl.maximum(max_val, sm)

    # store
    ODOHOW = OD * OH * OW
    out_base = n * C * ODOHOW + od * OH * OW + oh * OW
    out_ptrs = out_ptr + out_base + ow_offs[None, :] + c_offs[:, None] * ODOHOW
    tl.store(out_ptrs, max_val, mask=mask_2d)


def fused_softmax_pool(x, pool_total):
    N, C, D, H, W = x.shape
    OD = D // pool_total
    OH = H // pool_total
    OW = W // pool_total
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    BLOCK_OW = triton.next_power_of_2(OW)
    if BLOCK_OW < 8:
        BLOCK_OW = 8

    num_ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    grid = (N * OD * OH * num_ow_blocks,)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        POOL=pool_total,
        BLOCK_C=BLOCK_C,
        BLOCK_OW=BLOCK_OW,
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