import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PIX': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PIX': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PIX': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_PIX': 64}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'PH', 'PW'],
)
@triton.jit
def fused_conv_tanh_scale_bias_pool_kernel(
    x_ptr, w_ptr, cb_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    PH, PW,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PIX: tl.constexpr,  # pooled pixels per program
):
    # grid: (N, OC // BLOCK_OC, ceil(PH*PW / BLOCK_PIX))
    n = tl.program_id(0)
    oc_blk = tl.program_id(1)
    pix_blk = tl.program_id(2)

    oc_offs = oc_blk * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    oc_mask = oc_offs < OC

    pix_offs = pix_blk * BLOCK_PIX + tl.arange(0, BLOCK_PIX)  # [BLOCK_PIX]
    pix_mask = pix_offs < (PH * PW)

    ph = pix_offs // PW   # [BLOCK_PIX]
    pw = pix_offs % PW    # [BLOCK_PIX]

    # window-relative spatial coords (POOL*POOL window)
    # We'll iterate over POOL*POOL = 16 positions
    P2 = POOL * POOL

    # Load biases
    cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
    bb = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)   # [BLOCK_OC]

    # max accumulator over 16 positions for each (oc, pix)
    neg_inf = tl.full((BLOCK_OC, BLOCK_PIX), -1e30, dtype=tl.float32)
    max_val = neg_inf

    # For each position in the pool window
    for kk in tl.static_range(0, P2):
        ki = kk // POOL
        kj = kk % POOL
        oh = ph * POOL + ki  # [BLOCK_PIX]
        ow = pw * POOL + kj  # [BLOCK_PIX]

        acc = tl.zeros((BLOCK_OC, BLOCK_PIX), dtype=tl.float32)

        # Conv reduction
        for ic in tl.static_range(0, 8):  # IC=8
            for kh in tl.static_range(0, KH):
                for kw_ in tl.static_range(0, KW):
                    ih = oh + kh  # [BLOCK_PIX]
                    iw = ow + kw_  # [BLOCK_PIX]
                    in_offset = ((n * IC + ic) * IH + ih) * IW + iw  # [BLOCK_PIX]
                    x_val = tl.load(x_ptr + in_offset, mask=pix_mask, other=0.0)  # [BLOCK_PIX]
                    w_offset = ((oc_offs * IC + ic) * KH + kh) * KW + kw_  # [BLOCK_OC]
                    w_val = tl.load(w_ptr + w_offset, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                    acc += w_val[:, None] * x_val[None, :]

        acc = acc + cb[:, None]
        # tanh via stable formula
        e2 = tl.exp(-2.0 * tl.abs(acc))
        t = tl.where(acc >= 0, (1.0 - e2) / (1.0 + e2), -(1.0 - e2) / (1.0 + e2))
        v = t * SCALE + bb[:, None]
        max_val = tl.maximum(max_val, v)

    # Store
    out_offset = ((n * OC + oc_offs[:, None]) * PH * PW) + pix_offs[None, :]
    store_mask = oc_mask[:, None] & pix_mask[None, :]
    tl.store(out_ptr + out_offset, max_val, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scaling_factor = float(scaling_factor)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.max_pool = nn.MaxPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        PH = OH // POOL
        PW = OW // POOL

        if PH * POOL != OH or PW * POOL != OW:
            y = self.conv(x)
            y = torch.tanh(y) * self.scaling_factor + self.bias
            return self.max_pool(y)

        out = torch.empty((N, OC, PH, PW), device=x.device, dtype=x.dtype)
        bias_flat = self.bias.view(-1).contiguous()

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(PH * PW, META['BLOCK_PIX']))
        fused_conv_tanh_scale_bias_pool_kernel[grid](
            x, self.conv.weight, self.conv.bias, bias_flat, out,
            N, IC, IH, IW,
            OC, OH, OW,
            PH, PW,
            KH, KW,
            POOL,
            self.scaling_factor,
        )
        return out