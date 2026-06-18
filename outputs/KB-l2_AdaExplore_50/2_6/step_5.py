import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 4}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_OW': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OW': 8}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_OW': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OW': 8}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_OW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32}, num_warps=8, num_stages=2),
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
    # one program handles BLOCK_OW outputs along OW for one (n, od, oh)
    pid = tl.program_id(0)
    ow_blocks = (OW + BLOCK_OW - 1) // BLOCK_OW
    ow_blk = pid % ow_blocks
    tmp = pid // ow_blocks
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    ow_start = ow_blk * BLOCK_OW
    ow_offs = ow_start + tl.arange(0, BLOCK_OW)
    ow_mask = ow_offs < OW

    d0 = od * POOL
    h0 = oh * POOL
    w0 = ow_offs * POOL  # shape [BLOCK_OW]

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    DHW = D * H * W
    n_base = n * C * DHW
    c_stride = c_offs * DHW  # [BLOCK_C]

    NEG_INF = float('-inf')

    # max accumulator: [BLOCK_C, BLOCK_OW]
    max_val = tl.zeros([BLOCK_C, BLOCK_OW], dtype=tl.float32) + NEG_INF

    for dd in tl.static_range(0, POOL):
        d = d0 + dd
        d_base = d * H * W
        for hh in tl.static_range(0, POOL):
            h = h0 + hh
            h_base = h * W
            for ww in tl.static_range(0, POOL):
                w = w0 + ww  # [BLOCK_OW]
                # offsets shape [BLOCK_C, BLOCK_OW]
                base = n_base + d_base + h_base + w[None, :]
                ptrs = in_ptr + base + c_stride[:, None]
                full_mask = c_mask[:, None] & ow_mask[None, :]
                vals = tl.load(ptrs, mask=full_mask, other=NEG_INF)
                # softmax along C (axis=0)
                m = tl.max(vals, axis=0)  # [BLOCK_OW]
                ex = tl.exp(vals - m[None, :])
                ex = tl.where(c_mask[:, None], ex, 0.0)
                s = tl.sum(ex, axis=0)  # [BLOCK_OW]
                sm = ex / s[None, :]
                sm = tl.where(full_mask, sm, NEG_INF)
                max_val = tl.maximum(max_val, sm)

    # store: out shape (N, C, OD, OH, OW)
    ODOHOW = OD * OH * OW
    out_base = n * C * ODOHOW + od * OH * OW + oh * OW + ow_offs[None, :]
    out_ptrs = out_ptr + out_base + (c_offs * ODOHOW)[:, None]
    out_mask = c_mask[:, None] & ow_mask[None, :]
    tl.store(out_ptrs, max_val, mask=out_mask)


def fused_softmax_pool(x, pool_total):
    N, C, D, H, W = x.shape
    OD = D // pool_total
    OH = H // pool_total
    OW = W // pool_total
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    grid = lambda meta: (N * OD * OH * ((OW + meta['BLOCK_OW'] - 1) // meta['BLOCK_OW']),)
    fused_softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        POOL=pool_total,
        BLOCK_C=BLOCK_C,
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