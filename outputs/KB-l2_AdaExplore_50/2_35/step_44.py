import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 32, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128, 'BLOCK_IC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128, 'BLOCK_IC': 32}, num_warps=8, num_stages=2),
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

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    ic_range = tl.arange(0, BLOCK_IC)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    acc00 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Base offsets for input (NHWC) - for the (0,0) pool position with kh=0, kw=0
    base_h = poh * POOL_K
    base_w = pow_ * POOL_K
    n_base = pid_n * (H * W * IC)

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            w_base = oc_offs * (KH * KW * IC) + kh * (KW * IC) + kw * IC

            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + ic_range
                ic_mask = ic_idx < IC

                w_ptrs = w_ptr + w_base[:, None] + ic_idx[None, :]
                w_m = oc_mask[:, None] & ic_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_m, other=0.0)

                x_m = sp_mask[:, None] & ic_mask[None, :]

                # (0, 0)
                ih = base_h + kh
                iw = base_w + kw
                xb = n_base + ih * (W * IC) + iw * IC
                x00 = tl.load(x_ptr + xb[:, None] + ic_idx[None, :], mask=x_m, other=0.0)
                acc00 += tl.dot(w_vals, tl.trans(x00))

                # (0, 1)
                iw = base_w + 1 + kw
                xb = n_base + ih * (W * IC) + iw * IC
                x01 = tl.load(x_ptr + xb[:, None] + ic_idx[None, :], mask=x_m, other=0.0)
                acc01 += tl.dot(w_vals, tl.trans(x01))

                # (1, 1)
                ih = base_h + 1 + kh
                xb = n_base + ih * (W * IC) + iw * IC
                x11 = tl.load(x_ptr + xb[:, None] + ic_idx[None, :], mask=x_m, other=0.0)
                acc11 += tl.dot(w_vals, tl.trans(x11))

                # (1, 0)
                iw = base_w + kw
                xb = n_base + ih * (W * IC) + iw * IC
                x10 = tl.load(x_ptr + xb[:, None] + ic_idx[None, :], mask=x_m, other=0.0)
                acc10 += tl.dot(w_vals, tl.trans(x10))

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
        if (self._cached_w is None
                or self._cached_w.device != device):
            self._cached_w = self.conv.weight.detach().permute(0, 2, 3, 1).contiguous().to(device)
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