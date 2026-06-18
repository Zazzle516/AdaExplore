import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
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
    # x is NHWC: [N, H, W, IC], w is [OC, KH, KW, IC] (reordered), b is [OC]
    # out is [N, OC, POH, POW]
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

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Iterate pool window: each (pi, pj) corresponds to one OH×OW conv output
    # We compute it once and fold into max. No recomputation across pool positions.
    for pi in tl.static_range(0, POOL_K):
        for pj in tl.static_range(0, POOL_K):
            oh = poh * POOL_K + pi  # [BLOCK_SP]
            ow = pow_ * POOL_K + pj  # [BLOCK_SP]

            acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh  # [BLOCK_SP]
                    iw = ow + kw  # [BLOCK_SP]

                    # base offset into x for each sp: n*H*W*IC + ih*W*IC + iw*IC
                    x_base = pid_n * (H * W * IC) + ih * (W * IC) + iw * IC  # [BLOCK_SP]
                    # weight base: oc*KH*KW*IC + kh*KW*IC + kw*IC
                    w_base = oc_offs * (KH * KW * IC) + kh * (KW * IC) + kw * IC  # [BLOCK_OC]

                    for ic_start in range(0, IC, BLOCK_IC):
                        ic_idx = ic_start + ic_range  # [BLOCK_IC]
                        ic_mask = ic_idx < IC

                        # w[oc, kh, kw, ic]: shape [BLOCK_OC, BLOCK_IC]
                        w_ptrs = w_ptr + w_base[:, None] + ic_idx[None, :]
                        w_m = oc_mask[:, None] & ic_mask[None, :]
                        w_vals = tl.load(w_ptrs, mask=w_m, other=0.0)

                        # x[n, ih, iw, ic]: shape [BLOCK_SP, BLOCK_IC]
                        x_ptrs = x_ptr + x_base[:, None] + ic_idx[None, :]
                        x_m = sp_mask[:, None] & ic_mask[None, :]
                        x_vals = tl.load(x_ptrs, mask=x_m, other=0.0)

                        acc += tl.dot(w_vals, tl.trans(x_vals))

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
        x = x.cuda()
        # Convert x from NCHW to NHWC contiguous
        N, IC, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # Reorder weight from [OC, IC, KH, KW] to [OC, KH, KW, IC] contiguous
        w_orig = self.conv.weight  # [OC, IC, KH, KW]
        w = w_orig.permute(0, 2, 3, 1).contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        OC, KH, KW, _ = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        POOL_K = self.pool_kernel_size
        POH = OH // POOL_K
        POW = OW // POOL_K

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(POH * POW, meta['BLOCK_SP']),
        )

        fused_conv_pool_kernel[grid](
            x_nhwc, w, b, out,
            N, IC, H, W,
            OC, OH, OW,
            POH, POW,
            self.subtract_value,
            KH, KW,
            POOL_K,
        )

        return out