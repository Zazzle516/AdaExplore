import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 32, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
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
    # POOL_K is assumed to be 2 (hardcoded 4-accumulator unrolled pool).
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)  # [BLOCK_SP]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    ic_range = tl.arange(0, BLOCK_IC)  # [BLOCK_IC]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Four conv-output accumulators, one per pool position (POOL_K=2).
    acc00 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Hoisted: weight tile depends only on (kh, kw, ic_start), not on (pi, pj).
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            w_base = oc_offs * (KH * KW * IC) + kh * (KW * IC) + kw * IC  # [BLOCK_OC]

            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + ic_range  # [BLOCK_IC]
                ic_mask = ic_idx < IC

                # Load weight tile once and reuse across 4 pool positions.
                w_ptrs = w_ptr + w_base[:, None] + ic_idx[None, :]
                w_m = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_m, other=0.0)

                # Pool position (0, 0)
                ih00 = poh * POOL_K + 0 + kh
                iw00 = pow_ * POOL_K + 0 + kw
                x_base00 = pid_n * (H * W * IC) + ih00 * (W * IC) + iw00 * IC
                x_ptrs00 = x_ptr + x_base00[:, None] + ic_idx[None, :]
                x_m = sp_mask[:, None] & ic_mask[None, :]
                x00 = tl.load(x_ptrs00, mask=x_m, other=0.0)
                acc00 += tl.dot(w_vals, tl.trans(x00))

                # Pool position (0, 1)
                ih01 = poh * POOL_K + 0 + kh
                iw01 = pow_ * POOL_K + 1 + kw
                x_base01 = pid_n * (H * W * IC) + ih01 * (W * IC) + iw01 * IC
                x_ptrs01 = x_ptr + x_base01[:, None] + ic_idx[None, :]
                x01 = tl.load(x_ptrs01, mask=x_m, other=0.0)
                acc01 += tl.dot(w_vals, tl.trans(x01))

                # Pool position (1, 0)
                ih10 = poh * POOL_K + 1 + kh
                iw10 = pow_ * POOL_K + 0 + kw
                x_base10 = pid_n * (H * W * IC) + ih10 * (W * IC) + iw10 * IC
                x_ptrs10 = x_ptr + x_base10[:, None] + ic_idx[None, :]
                x10 = tl.load(x_ptrs10, mask=x_m, other=0.0)
                acc10 += tl.dot(w_vals, tl.trans(x10))

                # Pool position (1, 1)
                ih11 = poh * POOL_K + 1 + kh
                iw11 = pow_ * POOL_K + 1 + kw
                x_base11 = pid_n * (H * W * IC) + ih11 * (W * IC) + iw11 * IC
                x_ptrs11 = x_ptr + x_base11[:, None] + ic_idx[None, :]
                x11 = tl.load(x_ptrs11, mask=x_m, other=0.0)
                acc11 += tl.dot(w_vals, tl.trans(x11))

    # Apply bias - subtract value, then HardSwish.
    bias_col = bias[:, None]
    a00 = acc00 + bias_col - SUBV
    a01 = acc01 + bias_col - SUBV
    a10 = acc10 + bias_col - SUBV
    a11 = acc11 + bias_col - SUBV

    inv6 = 1.0 / 6.0
    hs00 = a00 * tl.minimum(tl.maximum(a00 + 3.0, 0.0), 6.0) * inv6
    hs01 = a01 * tl.minimum(tl.maximum(a01 + 3.0, 0.0), 6.0) * inv6
    hs10 = a10 * tl.minimum(tl.maximum(a10 + 3.0, 0.0), 6.0) * inv6
    hs11 = a11 * tl.minimum(tl.maximum(a11 + 3.0, 0.0), 6.0) * inv6

    # Reduce-max across pool window.
    m0 = tl.maximum(hs00, hs01)
    m1 = tl.maximum(hs10, hs11)
    max_acc = tl.maximum(m0, m1)

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
        self._cached_w = None
        self._cached_b = None

    def _get_packed_weight(self, device):
        # Lazily build [OC, KH, KW, IC] contiguous weight on the right device.
        w_param = self.conv.weight
        if (self._cached_w is None
                or self._cached_w.device != device
                or self._cached_w.data_ptr() == 0):
            self._cached_w = w_param.detach().permute(0, 2, 3, 1).contiguous().to(device)
        if (self._cached_b is None
                or self._cached_b.device != device):
            self._cached_b = self.conv.bias.detach().contiguous().to(device)
        return self._cached_w, self._cached_b

    def forward(self, x):
        x = x.cuda()
        N, IC, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w, b = self._get_packed_weight(x_nhwc.device)

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