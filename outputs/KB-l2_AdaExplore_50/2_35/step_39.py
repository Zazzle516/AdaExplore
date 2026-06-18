import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['OC', 'OUT_HW', 'IC_KHKW'],
)
@triton.jit
def conv_hs_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    POH, POW,
    OUT_HW,            # POH*POW (post-pool spatial)
    IC_KHKW,           # IC*KH*KW
    SUBV,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,   # OC tile
    BLOCK_N: tl.constexpr,   # post-pool spatial tile
    BLOCK_K: tl.constexpr,   # K reduction tile
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

    POOL_AREA: tl.constexpr = POOL_K * POOL_K

    # Accumulators for each pool position
    # We'll keep them as a [POOL_AREA, BLOCK_M, BLOCK_N] but Triton doesn't do 3D dot well;
    # instead unroll pool positions and keep separate accumulators.
    # Use a list-like approach via static_range and stack.

    # Initialize POOL_AREA accumulators
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Single K loop: K = IC*KH*KW
    # For each pool position (pi, pj), the input coords are:
    #   ih = poh*POOL_K + pi + kh
    #   iw = pow_*POOL_K + pj + kw
    # So we share weight loads across the 4 pool positions.

    for k_start in range(0, IC_KHKW, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IC_KHKW

        ic_idx = k_offs // (KH * KW)
        rem = k_offs % (KH * KW)
        kh_idx = rem // KW
        kw_idx = rem % KW

        # weight: [BLOCK_M, BLOCK_K]
        w_ptrs = w_ptr + (oc_offs[:, None] * IC_KHKW
                          + k_offs[None, :])
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # For each pool position, gather x and accumulate
        # pool position (0,0)
        oh0 = poh * POOL_K + 0
        ow0 = pow_ * POOL_K + 0
        ih0 = oh0[:, None] + kh_idx[None, :]
        iw0 = ow0[:, None] + kw_idx[None, :]
        x_ptrs0 = x_ptr + (pid_n * (IC * H * W)
                           + ic_idx[None, :] * (H * W)
                           + ih0 * W + iw0)
        x_mask0 = sp_mask[:, None] & k_mask[None, :] & (ih0 < H) & (iw0 < W)
        x_vals0 = tl.load(x_ptrs0, mask=x_mask0, other=0.0)
        acc0 += tl.dot(w_vals, tl.trans(x_vals0))

        # pool (0,1)
        oh1 = poh * POOL_K + 0
        ow1 = pow_ * POOL_K + 1
        ih1 = oh1[:, None] + kh_idx[None, :]
        iw1 = ow1[:, None] + kw_idx[None, :]
        x_ptrs1 = x_ptr + (pid_n * (IC * H * W)
                           + ic_idx[None, :] * (H * W)
                           + ih1 * W + iw1)
        x_mask1 = sp_mask[:, None] & k_mask[None, :] & (ih1 < H) & (iw1 < W)
        x_vals1 = tl.load(x_ptrs1, mask=x_mask1, other=0.0)
        acc1 += tl.dot(w_vals, tl.trans(x_vals1))

        # pool (1,0)
        oh2 = poh * POOL_K + 1
        ow2 = pow_ * POOL_K + 0
        ih2 = oh2[:, None] + kh_idx[None, :]
        iw2 = ow2[:, None] + kw_idx[None, :]
        x_ptrs2 = x_ptr + (pid_n * (IC * H * W)
                           + ic_idx[None, :] * (H * W)
                           + ih2 * W + iw2)
        x_mask2 = sp_mask[:, None] & k_mask[None, :] & (ih2 < H) & (iw2 < W)
        x_vals2 = tl.load(x_ptrs2, mask=x_mask2, other=0.0)
        acc2 += tl.dot(w_vals, tl.trans(x_vals2))

        # pool (1,1)
        oh3 = poh * POOL_K + 1
        ow3 = pow_ * POOL_K + 1
        ih3 = oh3[:, None] + kh_idx[None, :]
        iw3 = ow3[:, None] + kw_idx[None, :]
        x_ptrs3 = x_ptr + (pid_n * (IC * H * W)
                           + ic_idx[None, :] * (H * W)
                           + ih3 * W + iw3)
        x_mask3 = sp_mask[:, None] & k_mask[None, :] & (ih3 < H) & (iw3 < W)
        x_vals3 = tl.load(x_ptrs3, mask=x_mask3, other=0.0)
        acc3 += tl.dot(w_vals, tl.trans(x_vals3))

    # bias load
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # apply bias and subtract value, then hardswish, then max-pool, then mish
    def_b = bias[:, None] - SUBV

    a0 = acc0 + def_b
    a1 = acc1 + def_b
    a2 = acc2 + def_b
    a3 = acc3 + def_b

    # hardswish
    s0 = tl.minimum(tl.maximum(a0 + 3.0, 0.0), 6.0)
    s1 = tl.minimum(tl.maximum(a1 + 3.0, 0.0), 6.0)
    s2 = tl.minimum(tl.maximum(a2 + 3.0, 0.0), 6.0)
    s3 = tl.minimum(tl.maximum(a3 + 3.0, 0.0), 6.0)
    h0 = a0 * s0 * (1.0 / 6.0)
    h1 = a1 * s1 * (1.0 / 6.0)
    h2 = a2 * s2 * (1.0 / 6.0)
    h3 = a3 * s3 * (1.0 / 6.0)

    m01 = tl.maximum(h0, h1)
    m23 = tl.maximum(h2, h3)
    mx = tl.maximum(m01, m23)

    # mish
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