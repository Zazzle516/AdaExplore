import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OH_OW', 'IC_KHKW'],
)
@triton.jit
def conv_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    OH_OW,           # OH*OW
    IC_KHKW,         # IC*KH*KW
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
    sp_mask = sp_offs < OH_OW

    oh = sp_offs // OW
    ow = sp_offs % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KHKW = KH * KW

    for k_start in range(0, IC_KHKW, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IC_KHKW

        ic_idx = k_offs // KHKW
        rem = k_offs % KHKW
        kh_idx = rem // KW
        kw_idx = rem % KW

        # weight: [OC, IC, KH, KW]
        w_ptrs = w_ptr + (oc_offs[:, None] * IC_KHKW + k_offs[None, :])
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        ih = oh[:, None] + kh_idx[None, :]
        iw = ow[:, None] + kw_idx[None, :]
        x_ptrs = x_ptr + (pid_n * (IC * H * W)
                          + ic_idx[None, :] * (H * W)
                          + ih * W
                          + iw)
        x_mask = sp_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, tl.trans(x_vals))

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None] - SUBV

    # hardswish
    shifted = acc + 3.0
    shifted = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    hs = acc * shifted * (1.0 / 6.0)

    out_ptrs = out_ptr + (pid_n * (OC * OH_OW)
                          + oc_offs[:, None] * OH_OW
                          + sp_offs[None, :])
    store_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptrs, hs, mask=store_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
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

    # stable softplus + mish: tanh(softplus(x))
    # softplus(x) = x for large x, else log1p(exp(x))
    sp_val = tl.where(max_v > 20.0, max_v, tl.log(1.0 + tl.exp(max_v)))
    e2 = tl.exp(-2.0 * sp_val)
    tanh_sp = (1.0 - e2) / (1.0 + e2)
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
        OH_OW = OH * OW
        IC_KHKW = IC * KH * KW

        conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        grid_conv = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OH_OW, meta['BLOCK_N']),
        )

        conv_kernel[grid_conv](
            x, w, b, conv_out,
            N, IC, H, W,
            OC, OH, OW,
            OH_OW, IC_KHKW,
            self.subtract_value,
            KH, KW,
        )

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)
        TOTAL = N * OC * POH * POW
        grid_pool = lambda meta: (triton.cdiv(TOTAL, meta['BLOCK']),)
        pool_mish_kernel[grid_pool](
            conv_out, out,
            N, OC, OH, OW, POH, POW, POOL_K,
            TOTAL,
        )

        return out