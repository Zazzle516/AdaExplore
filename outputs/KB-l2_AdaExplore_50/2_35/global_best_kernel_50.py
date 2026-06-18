import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _hardswish(v):
    s = v + 3.0
    s = tl.minimum(tl.maximum(s, 0.0), 6.0)
    return v * s * (1.0 / 6.0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'POH_POW', 'IC_KHKW'],
)
@triton.jit
def fused_conv_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW, POH, POW,
    POH_POW,
    IC_KHKW,
    SUBV,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)
    sp_offs = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < POH_POW

    ph = sp_offs // POW
    pw = sp_offs % POW
    oh_base = ph * 2
    ow_base = pw * 2

    acc00 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHKW = KH * KW
    HW = H * W
    x_batch_base = pid_n * IC * HW

    for k_start in range(0, IC_KHKW, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IC_KHKW

        ic_idx = k_offs // KHKW
        rem = k_offs % KHKW
        kh_idx = rem // KW
        kw_idx = rem % KW

        w_ptrs = w_ptr + (oc_offs[:, None] * IC_KHKW + k_offs[None, :])
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        x_mask = sp_mask[:, None] & k_mask[None, :]
        ic_hw = ic_idx[None, :] * HW
        kh_w = kh_idx[None, :] * W

        # di=0, dj=0
        ih = oh_base[:, None] + kh_w
        iw = ow_base[:, None] + kw_idx[None, :]
        x_ptrs = x_ptr + x_batch_base + ic_hw + ih * W + iw
        # Wait: ih already incorporates *W via kh_w. Recompute below.

        # Correct addressing: in_index = ic * HW + (oh+kh) * W + (ow+kw)
        # = ic*HW + oh*W + kh*W + ow + kw
        base_a = x_batch_base + ic_hw + kh_w + kw_idx[None, :]
        row00 = (oh_base[:, None]) * W + ow_base[:, None]
        x_ptrs = x_ptr + base_a + row00
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        acc00 += tl.dot(w_vals, tl.trans(x_vals))

        # di=0, dj=1
        row01 = (oh_base[:, None]) * W + (ow_base[:, None] + 1)
        x_ptrs = x_ptr + base_a + row01
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        acc01 += tl.dot(w_vals, tl.trans(x_vals))

        # di=1, dj=0
        row10 = (oh_base[:, None] + 1) * W + ow_base[:, None]
        x_ptrs = x_ptr + base_a + row10
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        acc10 += tl.dot(w_vals, tl.trans(x_vals))

        # di=1, dj=1
        row11 = (oh_base[:, None] + 1) * W + (ow_base[:, None] + 1)
        x_ptrs = x_ptr + base_a + row11
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)
        acc11 += tl.dot(w_vals, tl.trans(x_vals))

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    bias_col = bias[:, None] - SUBV

    v00 = _hardswish(acc00 + bias_col)
    v01 = _hardswish(acc01 + bias_col)
    v10 = _hardswish(acc10 + bias_col)
    v11 = _hardswish(acc11 + bias_col)

    m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    # mish(m) = m * tanh(softplus(m))
    sp = tl.log(1.0 + tl.exp(m))
    e2 = tl.exp(2.0 * sp)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out_val = m * tanh_sp

    out_ptrs = out_ptr + (pid_n * OC * POH_POW
                          + oc_offs[:, None] * POH_POW
                          + sp_offs[None, :])
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptrs, out_val, mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
    ],
    key=['TOTAL'],
)
@triton.jit
def pool_mish_kernel(
    in_ptr, out_ptr,
    N, OC, OH, OW, POH, POW, POOL_K: tl.constexpr,
    TOTAL,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL

    pw = offs % POW
    tmp = offs // POW
    ph = tmp % POH
    tmp2 = tmp // POH
    oc = tmp2 % OC
    n = tmp2 // OC

    base_in = n * (OC * OH * OW) + oc * (OH * OW)
    h0 = ph * POOL_K
    w0 = pw * POOL_K

    NEG_INF = float('-inf')
    max_v = tl.full((BLOCK,), NEG_INF, dtype=tl.float32)

    for i in tl.static_range(0, POOL_K):
        for j in tl.static_range(0, POOL_K):
            ih = h0 + i
            iw = w0 + j
            ptrs = in_ptr + base_in + ih * OW + iw
            v = tl.load(ptrs, mask=mask, other=NEG_INF)
            max_v = tl.maximum(max_v, v)

    sp_val = tl.log(1.0 + tl.exp(max_v))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = max_v * tanh_sp

    tl.store(out_ptr + offs, out, mask=mask)


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
        POH_POW = POH * POW
        IC_KHKW = IC * KH * KW

        if POOL_K == 2:
            out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)
            w_flat = w.view(OC, IC_KHKW).contiguous()
            grid = lambda meta: (
                N,
                triton.cdiv(OC, meta['BLOCK_M']),
                triton.cdiv(POH_POW, meta['BLOCK_N']),
            )
            fused_conv_pool_mish_kernel[grid](
                x, w_flat, b, out,
                N, IC, H, W,
                OC, OH, OW, POH, POW,
                POH_POW, IC_KHKW,
                self.subtract_value,
                KH, KW,
            )
            return out
        else:
            # Fallback to PyTorch for unsupported pool sizes
            y = F.conv2d(x, w, b) - self.subtract_value
            y = F.hardswish(y)
            y = F.max_pool2d(y, POOL_K)
            return F.mish(y)