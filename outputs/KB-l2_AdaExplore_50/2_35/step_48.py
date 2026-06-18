import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'POOLED_HW', 'IC', 'KH', 'KW'],
)
@triton.jit
def fused_conv_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    POH, POW,
    POOL_K: tl.constexpr,
    SUBV,
    POOLED_HW,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    oc_mask = oc_offs < OC
    sp_mask = sp_offs < POOLED_HW

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    NEG_INF = float('-inf')
    # 4 accumulators for the 2x2 pool window (POOL_K=2 assumed/general up to small)
    # We'll use a static_range loop for pool window but accumulate all positions in parallel
    # by storing them separately.

    # max_acc tracks running max of hardswish over pool window
    max_acc = tl.full((BLOCK_OC, BLOCK_SP), NEG_INF, dtype=tl.float32)

    # Build acc tensors for each pool position. POOL_K is small (typically 2).
    # We compute conv outputs for all POOL_K*POOL_K positions sharing weight loads.
    # Initialize separate accumulators per pool position.

    # We'll loop over (pi, pj) with static_range and within K-loop share weight load.
    # To share weight loads, we put pool window iteration *inside* K-loop body.
    # That requires keeping POOL_K^2 accumulators.

    acc00 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # base output coords for pool position (0,0), (0,1), (1,0), (1,1)
    oh_base = poh * POOL_K   # [BLOCK_SP]
    ow_base = pow_ * POOL_K  # [BLOCK_SP]

    # K-loop: iterate over (kh, kw) statically and over IC in chunks
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            for ic_start in range(0, IC, BLOCK_IC):
                ic_offs = ic_start + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # weight load: w[oc, ic, kh, kw]
                w_ptrs = w_ptr + (oc_offs[:, None] * (IC * KH * KW)
                                  + ic_offs[None, :] * (KH * KW)
                                  + kh * KW + kw)
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_IC]

                # For each pool position, compute the input gather and FMA
                # pool pos (0,0): oh = oh_base+0, ow = ow_base+0
                ih00 = oh_base + 0 + kh
                iw00 = ow_base + 0 + kw
                x_ptrs00 = x_ptr + (pid_n * (IC * H * W)
                                    + ic_offs[None, :] * (H * W)
                                    + ih00[:, None] * W
                                    + iw00[:, None])
                x_mask00 = sp_mask[:, None] & ic_mask[None, :] & (ih00[:, None] < H) & (iw00[:, None] < W)
                x_vals00 = tl.load(x_ptrs00, mask=x_mask00, other=0.0)
                acc00 += tl.dot(w_vals, tl.trans(x_vals00))

                # pool pos (0,1)
                if POOL_K > 1:
                    ih01 = oh_base + 0 + kh
                    iw01 = ow_base + 1 + kw
                    x_ptrs01 = x_ptr + (pid_n * (IC * H * W)
                                        + ic_offs[None, :] * (H * W)
                                        + ih01[:, None] * W
                                        + iw01[:, None])
                    x_mask01 = sp_mask[:, None] & ic_mask[None, :] & (ih01[:, None] < H) & (iw01[:, None] < W)
                    x_vals01 = tl.load(x_ptrs01, mask=x_mask01, other=0.0)
                    acc01 += tl.dot(w_vals, tl.trans(x_vals01))

                    # pool pos (1,0)
                    ih10 = oh_base + 1 + kh
                    iw10 = ow_base + 0 + kw
                    x_ptrs10 = x_ptr + (pid_n * (IC * H * W)
                                        + ic_offs[None, :] * (H * W)
                                        + ih10[:, None] * W
                                        + iw10[:, None])
                    x_mask10 = sp_mask[:, None] & ic_mask[None, :] & (ih10[:, None] < H) & (iw10[:, None] < W)
                    x_vals10 = tl.load(x_ptrs10, mask=x_mask10, other=0.0)
                    acc10 += tl.dot(w_vals, tl.trans(x_vals10))

                    # pool pos (1,1)
                    ih11 = oh_base + 1 + kh
                    iw11 = ow_base + 1 + kw
                    x_ptrs11 = x_ptr + (pid_n * (IC * H * W)
                                        + ic_offs[None, :] * (H * W)
                                        + ih11[:, None] * W
                                        + iw11[:, None])
                    x_mask11 = sp_mask[:, None] & ic_mask[None, :] & (ih11[:, None] < H) & (iw11[:, None] < W)
                    x_vals11 = tl.load(x_ptrs11, mask=x_mask11, other=0.0)
                    acc11 += tl.dot(w_vals, tl.trans(x_vals11))

    # bias
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc00 = acc00 + bias[:, None] - SUBV
    acc01 = acc01 + bias[:, None] - SUBV
    acc10 = acc10 + bias[:, None] - SUBV
    acc11 = acc11 + bias[:, None] - SUBV

    # hardswish on each
    def _hs(v):
        s = tl.minimum(tl.maximum(v + 3.0, 0.0), 6.0)
        return v * s * (1.0 / 6.0)

    hs00 = _hs(acc00)
    hs01 = _hs(acc01)
    hs10 = _hs(acc10)
    hs11 = _hs(acc11)

    # max-reduce
    m = tl.maximum(hs00, hs01)
    m = tl.maximum(m, hs10)
    m = tl.maximum(m, hs11)

    # mish: x * tanh(softplus(x))
    sp_val = tl.log(1.0 + tl.exp(m))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = m * tanh_sp

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

        # Choose BLOCK_IC: prefer power-of-2 >= IC if IC small, else 32/64
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
            triton.cdiv(POOLED_HW, meta['BLOCK_SP']),
        )

        fused_conv_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            POH, POW,
            POOL_K,
            self.subtract_value,
            POOLED_HW,
            BLOCK_IC=BLOCK_IC,
        )

        return out