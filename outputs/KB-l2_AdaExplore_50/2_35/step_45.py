import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'POOLED_HW', 'IC_KHKW'],
)
@triton.jit
def fused_conv_subv_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH: tl.constexpr, KW: tl.constexpr,
    OH, OW,
    POH, POW,
    POOL_K: tl.constexpr,
    SUBV,
    POOLED_HW,
    IC_KHKW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    POOL_K2: tl.constexpr = POOL_K * POOL_K
    BN_FULL: tl.constexpr = BLOCK_N * POOL_K2

    oc_offs = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)
    sp_offs = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < POOLED_HW

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    pp = tl.arange(0, POOL_K2)
    pi = pp // POOL_K
    pj = pp % POOL_K

    oh2 = poh[:, None] * POOL_K + pi[None, :]
    ow2 = pow_[:, None] * POOL_K + pj[None, :]
    valid_sp2 = sp_mask[:, None] & (oh2 < OH) & (ow2 < OW)

    oh_flat = tl.reshape(oh2, (BN_FULL,))
    ow_flat = tl.reshape(ow2, (BN_FULL,))
    valid_flat = tl.reshape(valid_sp2, (BN_FULL,))

    acc = tl.zeros((BLOCK_M, BN_FULL), dtype=tl.float32)

    for k_start in range(0, IC_KHKW, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IC_KHKW

        ic_idx = k_offs // (KH * KW)
        rem = k_offs % (KH * KW)
        kh_idx = rem // KW
        kw_idx = rem % KW

        w_ptrs = w_ptr + (oc_offs[:, None] * (IC * KH * KW)
                          + ic_idx[None, :] * (KH * KW)
                          + kh_idx[None, :] * KW
                          + kw_idx[None, :])
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        ih = oh_flat[:, None] + kh_idx[None, :]
        iw = ow_flat[:, None] + kw_idx[None, :]
        x_ptrs = x_ptr + (pid_n * (IC * H * W)
                          + ic_idx[None, :] * (H * W)
                          + ih * W
                          + iw)
        x_mask = valid_flat[:, None] & k_mask[None, :] & (ih < H) & (iw < W)
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, tl.trans(x_vals))

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None] - SUBV

    shifted = acc + 3.0
    shifted = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
    hs = acc * shifted * (1.0 / 6.0)

    NEG_INF = float('-inf')
    hs = tl.where(valid_flat[None, :], hs, NEG_INF)

    hs3 = tl.reshape(hs, (BLOCK_M, BLOCK_N, POOL_K2))
    max_pooled = tl.max(hs3, axis=2)

    sp_val = tl.log(1.0 + tl.exp(max_pooled))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = max_pooled * tanh_sp

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
        IC_KHKW = IC * KH * KW

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(POOLED_HW, meta['BLOCK_N']),
        )

        fused_conv_subv_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            POH, POW,
            POOL_K,
            self.subtract_value,
            POOLED_HW,
            IC_KHKW,
        )

        return out