import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 128}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def conv_sub_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH_pre, OW_pre,
    OH, OW,
    SUB: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    oc_mask = oc_offs < OC
    hw_mask = hw_offs < (OH * OW)

    oh = hw_offs // OW
    ow = hw_offs % OW

    ih0 = oh * POOL
    iw0 = ow * POOL

    NEG_INF = float('-inf')
    max_val = tl.full((BLOCK_OC, BLOCK_HW), NEG_INF, dtype=tl.float32)

    # Load bias once
    b_v = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # For each pre-pool position in pool window (POOL*POOL of them)
    # We compute the conv output and fuse hardswish, then take max.
    # Pre-compute conv accumulators for each pool position - but we
    # interleave: for each pool position separately compute and update max.
    # To share input loads across pool positions, we can compute all
    # POOL*POOL conv outputs in registers simultaneously sharing weight loads.

    # Allocate POOL*POOL accumulators
    acc00 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Iterate over IC, KH, KW
    for ic in range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                w_off = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_v = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                # For each pool offset (ph, pw) compute the input position
                # ph=0, pw=0
                ih = ih0 + 0 + kh
                iw = iw0 + 0 + kw
                in_valid = hw_mask & (ih < IH) & (iw < IW)
                x_off = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_v = tl.load(x_ptr + x_off, mask=in_valid, other=0.0)
                acc00 += w_v[:, None] * x_v[None, :]

                # ph=0, pw=1
                ih = ih0 + 0 + kh
                iw = iw0 + 1 + kw
                in_valid = hw_mask & (ih < IH) & (iw < IW)
                x_off = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_v = tl.load(x_ptr + x_off, mask=in_valid, other=0.0)
                acc01 += w_v[:, None] * x_v[None, :]

                # ph=1, pw=0
                ih = ih0 + 1 + kh
                iw = iw0 + 0 + kw
                in_valid = hw_mask & (ih < IH) & (iw < IW)
                x_off = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_v = tl.load(x_ptr + x_off, mask=in_valid, other=0.0)
                acc10 += w_v[:, None] * x_v[None, :]

                # ph=1, pw=1
                ih = ih0 + 1 + kh
                iw = iw0 + 1 + kw
                in_valid = hw_mask & (ih < IH) & (iw < IW)
                x_off = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
                x_v = tl.load(x_ptr + x_off, mask=in_valid, other=0.0)
                acc11 += w_v[:, None] * x_v[None, :]

    # Apply bias, subtract, hardswish, take max (inlined, no nested functions)
    b_col = b_v[:, None]

    v00 = acc00 + b_col - SUB
    relu6_00 = tl.minimum(tl.maximum(v00 + 3.0, 0.0), 6.0)
    hs00 = v00 * relu6_00 * (1.0 / 6.0)

    v01 = acc01 + b_col - SUB
    relu6_01 = tl.minimum(tl.maximum(v01 + 3.0, 0.0), 6.0)
    hs01 = v01 * relu6_01 * (1.0 / 6.0)

    v10 = acc10 + b_col - SUB
    relu6_10 = tl.minimum(tl.maximum(v10 + 3.0, 0.0), 6.0)
    hs10 = v10 * relu6_10 * (1.0 / 6.0)

    v11 = acc11 + b_col - SUB
    relu6_11 = tl.minimum(tl.maximum(v11 + 3.0, 0.0), 6.0)
    hs11 = v11 * relu6_11 * (1.0 / 6.0)

    max_val = tl.maximum(tl.maximum(hs00, hs01), tl.maximum(hs10, hs11))

    # Mish: x * tanh(softplus(x)) — numerically stable
    sp = tl.log(1.0 + tl.exp(-tl.abs(max_val))) + tl.maximum(max_val, 0.0)
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    out = max_val * tanh_sp

    out_off = pid_n * OC * OH * OW + oc_offs[:, None] * (OH * OW) + hw_offs[None, :]
    store_mask = oc_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = pool_kernel_size
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH_pre = IH - KH + 1
        OW_pre = IW - KW + 1
        POOL = self.pool_kernel_size
        OH = OH_pre // POOL
        OW = OW_pre // POOL

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(OH * OW, meta['BLOCK_HW']),
        )

        conv_sub_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH_pre, OW_pre,
            OH, OW,
            self.subtract_value,
            KH, KW, POOL,
        )
        return out