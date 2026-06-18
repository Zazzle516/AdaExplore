import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KHKW'],
)
@triton.jit
def conv_hs_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    POH, POW,
    OUT_HW,
    IC_KHKW,
    SUBV,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL_K: tl.constexpr,
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
    sp_mask = sp_offs < OUT_HW

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # base output coords (top-left of pool window)
    oh_base = poh * POOL_K  # [BLOCK_N]
    ow_base = pow_ * POOL_K  # [BLOCK_N]

    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_n_base = pid_n * (IC * H * W)

    for k_start in range(0, IC_KHKW, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IC_KHKW

        ic_idx = k_offs // (KH * KW)
        rem = k_offs % (KH * KW)
        kh_idx = rem // KW
        kw_idx = rem % KW

        # weight: [BLOCK_M, BLOCK_K]
        w_ptrs = w_ptr + (oc_offs[:, None] * IC_KHKW + k_offs[None, :])
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        ic_hw = ic_idx * (H * W)  # [BLOCK_K]
        khw = kh_idx * W + kw_idx  # [BLOCK_K]

        # pool (0,0)
        base0 = oh_base * W + ow_base  # [BLOCK_N]
        x_ptrs0 = x_ptr + x_n_base + ic_hw[None, :] + base0[:, None] + khw[None, :]
        x_mask0 = sp_mask[:, None] & k_mask[None, :]
        x_vals0 = tl.load(x_ptrs0, mask=x_mask0, other=0.0)
        acc0 += tl.dot(w_vals, tl.trans(x_vals0))

        # pool (0,1)
        base1 = oh_base * W + ow_base + 1
        x_ptrs1 = x_ptr + x_n_base + ic_hw[None, :] + base1[:, None] + khw[None, :]
        x_vals1 = tl.load(x_ptrs1, mask=x_mask0, other=0.0)
        acc1 += tl.dot(w_vals, tl.trans(x_vals1))

        # pool (1,0)
        base2 = (oh_base + 1) * W + ow_base
        x_ptrs2 = x_ptr + x_n_base + ic_hw[None, :] + base2[:, None] + khw[None, :]
        x_vals2 = tl.load(x_ptrs2, mask=x_mask0, other=0.0)
        acc2 += tl.dot(w_vals, tl.trans(x_vals2))

        # pool (1,1)
        base3 = (oh_base + 1) * W + ow_base + 1
        x_ptrs3 = x_ptr + x_n_base + ic_hw[None, :] + base3[:, None] + khw[None, :]
        x_vals3 = tl.load(x_ptrs3, mask=x_mask0, other=0.0)
        acc3 += tl.dot(w_vals, tl.trans(x_vals3))

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    bb = bias[:, None] - SUBV

    a0 = acc0 + bb
    a1 = acc1 + bb
    a2 = acc2 + bb
    a3 = acc3 + bb

    s0 = tl.minimum(tl.maximum(a0 + 3.0, 0.0), 6.0)
    s1 = tl.minimum(tl.maximum(a1 + 3.0, 0.0), 6.0)
    s2 = tl.minimum(tl.maximum(a2 + 3.0, 0.0), 6.0)
    s3 = tl.minimum(tl.maximum(a3 + 3.0, 0.0), 6.0)
    inv6 = 1.0 / 6.0
    h0 = a0 * s0 * inv6
    h1 = a1 * s1 * inv6
    h2 = a2 * s2 * inv6
    h3 = a3 * s3 * inv6

    mx = tl.maximum(tl.maximum(h0, h1), tl.maximum(h2, h3))

    # mish: x * tanh(softplus(x))
    sp_v = tl.log(1.0 + tl.exp(mx))
    e2 = tl.exp(2.0 * sp_v)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = mx * tanh_sp

    out_ptrs = out_ptr + (pid_n * (OC * OUT_HW)
                          + oc_offs[:, None] * OUT_HW
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
        OUT_HW = POH * POW
        IC_KHKW = IC * KH * KW

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OUT_HW, meta['BLOCK_N']),
        )

        conv_hs_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, OH, OW,
            POH, POW,
            OUT_HW, IC_KHKW,
            self.subtract_value,
            KH, KW, POOL_K,
        )

        return out