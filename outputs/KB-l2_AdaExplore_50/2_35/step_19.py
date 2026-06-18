import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'POH', 'POW', 'IC'],
)
@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    POH, POW,
    SUBV,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    NEG_INF = float('-inf')
    max_acc = tl.full((BLOCK_OC, BLOCK_SP), NEG_INF, dtype=tl.float32)

    ic_range = tl.arange(0, BLOCK_IC)  # [BLOCK_IC]

    # Iterate pool window
    for pi in tl.static_range(0, POOL_K):
        for pj in tl.static_range(0, POOL_K):
            oh = poh * POOL_K + pi  # [BLOCK_SP]
            ow = pow_ * POOL_K + pj  # [BLOCK_SP]

            acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            # Loop over kh, kw (small, static), and IC chunks
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh  # [BLOCK_SP]
                    iw = ow + kw  # [BLOCK_SP]

                    for ic_start in range(0, IC, BLOCK_IC):
                        ic_idx = ic_start + ic_range  # [BLOCK_IC]
                        ic_mask = ic_idx < IC

                        # weight ptr: w[oc, ic, kh, kw]
                        w_ptrs = w_ptr + (oc_offs[:, None] * (IC * KH * KW)
                                          + ic_idx[None, :] * (KH * KW)
                                          + kh * KW + kw)
                        w_m = oc_mask[:, None] & ic_mask[None, :]
                        w_vals = tl.load(w_ptrs, mask=w_m, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                        # x ptr: x[n, ic, ih, iw]
                        x_ptrs = x_ptr + (pid_n * (IC * H * W)
                                          + ic_idx[None, :] * (H * W)
                                          + ih[:, None] * W
                                          + iw[:, None])
                        x_m = sp_mask[:, None] & ic_mask[None, :]
                        x_vals = tl.load(x_ptrs, mask=x_m, other=0.0)  # [BLOCK_SP, BLOCK_IC]

                        acc += tl.dot(w_vals, tl.trans(x_vals))

            bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
            acc = acc + bias[:, None] - SUBV

            # hardswish
            shifted = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
            hs = acc * shifted * (1.0 / 6.0)

            hs = tl.where(sp_mask[None, :], hs, NEG_INF)
            max_acc = tl.maximum(max_acc, hs)

    # mish: x * tanh(softplus(x))
    sp_val = tl.log(1.0 + tl.exp(max_acc))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = max_acc * tanh_sp

    out_ptrs = out_ptr + (pid_n * (OC * POH * POW)
                          + oc_offs[:, None] * (POH * POW)
                          + sp_offs[None, :])
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptrs, out, mask=store_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        POOL_K = self.pool_kernel_size
        POH = OH // POOL_K
        POW = OW // POOL_K

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        # Pick BLOCK_IC as power-of-two >= IC (or capped)
        if IC <= 16:
            BLOCK_IC = 16
        elif IC <= 32:
            BLOCK_IC = 32
        elif IC <= 64:
            BLOCK_IC = 64
        else:
            BLOCK_IC = 64

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(POH * POW, meta['BLOCK_SP']),
        )

        fused_conv_pool_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, OH, OW,
            POH, POW,
            self.subtract_value,
            KH, KW,
            POOL_K,
            BLOCK_IC=BLOCK_IC,
        )

        return out