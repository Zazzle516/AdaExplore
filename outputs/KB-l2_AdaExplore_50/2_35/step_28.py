import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'POOLED_HW', 'IC'],
)
@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, H, W,
    OC, OH, OW,
    POH, POW,
    POOLED_HW,
    SUBV,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < POOLED_HW

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    NEG_INF = float('-inf')
    max_acc = tl.full((BLOCK_OC, BLOCK_SP), NEG_INF, dtype=tl.float32)

    ic_range = tl.arange(0, IC)  # [IC]

    # Iterate over pool window
    for pi in tl.static_range(0, POOL_K):
        for pj in tl.static_range(0, POOL_K):
            oh = poh * POOL_K + pi    # [BLOCK_SP]
            ow = pow_ * POOL_K + pj   # [BLOCK_SP]
            valid_sp = (oh < OH) & (ow < OW) & sp_mask

            acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            # Loop over kernel positions; BLOCK_K = IC (channels contiguous after we load)
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    # weight: w[oc, ic, kh, kw], shape (OC, IC, KH, KW)
                    # Load [BLOCK_OC, IC] for this (kh, kw)
                    w_ptrs = w_ptr + (oc_offs[:, None] * (IC * KH * KW)
                                      + ic_range[None, :] * (KH * KW)
                                      + kh * KW + kw)
                    w_vals = tl.load(w_ptrs, mask=oc_mask[:, None], other=0.0)  # [BLOCK_OC, IC]

                    # input: x[n, ic, oh+kh, ow+kw] -> [BLOCK_SP, IC]
                    ih = oh + kh   # [BLOCK_SP]
                    iw = ow + kw   # [BLOCK_SP]
                    x_ptrs = x_ptr + (pid_n * (IC * H * W)
                                      + ic_range[None, :] * (H * W)
                                      + ih[:, None] * W
                                      + iw[:, None])
                    x_mask = valid_sp[:, None] & (ih < H)[:, None] & (iw < W)[:, None]
                    x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)  # [BLOCK_SP, IC]

                    # acc[BLOCK_OC, BLOCK_SP] += w @ x^T
                    acc += tl.dot(w_vals, tl.trans(x_vals))

            bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
            acc = acc + bias[:, None] - SUBV

            # hardswish
            shifted = acc + 3.0
            shifted = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
            hs = acc * shifted * (1.0 / 6.0)

            hs = tl.where(valid_sp[None, :], hs, NEG_INF)
            max_acc = tl.maximum(max_acc, hs)

    # Mish
    sp_val = tl.log(1.0 + tl.exp(max_acc))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = max_acc * tanh_sp

    out_ptrs = out_ptr + (pid_n * (OC * POOLED_HW)
                          + oc_offs[:, None] * POOLED_HW
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
        POOLED_HW = POH * POW

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(POOLED_HW, meta['BLOCK_SP']),
        )

        fused_conv_pool_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, OH, OW,
            POH, POW,
            POOLED_HW,
            self.subtract_value,
            KH, KW,
            POOL_K,
        )

        return out