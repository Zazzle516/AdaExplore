import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64, 'BLOCK_IC': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_HW': 32, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 128, 'BLOCK_IC': 16}, num_warps=8, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW'],
)
@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC,
    OH_pre, OW_pre,
    OH, OW,
    SUB: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
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
    b_v = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]

    # For each position in pool window, compute conv result and reduce.
    for ph in tl.static_range(0, POOL):
        for pw in tl.static_range(0, POOL):
            h_pre = ih0 + ph
            w_pre = iw0 + pw
            valid = hw_mask & (h_pre < OH_pre) & (w_pre < OW_pre)

            acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

            # Loop IC in blocks, KH/KW fully unrolled
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)  # [BLOCK_IC]
                ic_mask = ic_offs < IC

                for kh in tl.static_range(0, KH):
                    for kw in tl.static_range(0, KW):
                        ih = h_pre + kh  # [BLOCK_HW]
                        iw = w_pre + kw
                        # x offset: [BLOCK_IC, BLOCK_HW]
                        x_off = (pid_n * IC * IH * IW
                                 + ic_offs[:, None] * (IH * IW)
                                 + ih[None, :] * IW
                                 + iw[None, :])
                        x_mask = ic_mask[:, None] & valid[None, :]
                        x_v = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_IC, BLOCK_HW]

                        # w offset: [BLOCK_OC, BLOCK_IC]
                        w_off = (oc_offs[:, None] * (IC * KH * KW)
                                 + ic_offs[None, :] * (KH * KW)
                                 + kh * KW + kw)
                        w_mask = oc_mask[:, None] & ic_mask[None, :]
                        w_v = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                        acc += tl.dot(w_v, x_v, allow_tf32=False)

            acc = acc + b_v[:, None] - SUB

            # HardSwish
            xp3 = acc + 3.0
            relu6 = tl.minimum(tl.maximum(xp3, 0.0), 6.0)
            hs = acc * relu6 * (1.0 / 6.0)

            hs = tl.where(valid[None, :], hs, NEG_INF)
            max_val = tl.maximum(max_val, hs)

    # Mish
    sp = tl.log(1.0 + tl.exp(max_val))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = (e2 - 1.0) / (e2 + 1.0)
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

        fused_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC,
            OH_pre, OW_pre,
            OH, OW,
            self.subtract_value,
            KH, KW, POOL,
        )
        return out