import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'POOLED_HW', 'IC_KHKW'],
)
@triton.jit
def fused_conv_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    POH, POW,
    POOLED_HW,
    SUBV,
    KH: tl.constexpr, KW: tl.constexpr,
    POOL_K: tl.constexpr,
    POOL_AREA: tl.constexpr,   # POOL_K*POOL_K
    IC_KHKW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,     # number of pooled positions per tile
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)            # [BLOCK_M]
    sp_offs = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)            # [BLOCK_N] pooled spatial
    pa_offs = tl.arange(0, POOL_AREA)                              # [POOL_AREA]

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < POOLED_HW

    poh = sp_offs // POW       # [BLOCK_N]
    pow_ = sp_offs % POW       # [BLOCK_N]

    # full output positions (BLOCK_N, POOL_AREA)
    pi = pa_offs // POOL_K     # [POOL_AREA]
    pj = pa_offs % POOL_K      # [POOL_AREA]

    oh = poh[:, None] * POOL_K + pi[None, :]   # [BLOCK_N, POOL_AREA]
    ow = pow_[:, None] * POOL_K + pj[None, :]  # [BLOCK_N, POOL_AREA]

    # Flatten N-axis to BLOCK_N * POOL_AREA "columns"
    NCOLS: tl.constexpr = BLOCK_N * POOL_AREA
    oh_flat = tl.reshape(oh, (NCOLS,))
    ow_flat = tl.reshape(ow, (NCOLS,))

    valid_col = (oh_flat < OH) & (ow_flat < OW) & tl.reshape(
        tl.broadcast_to(sp_mask[:, None], (BLOCK_N, POOL_AREA)), (NCOLS,)
    )

    acc = tl.zeros((BLOCK_M, NCOLS), dtype=tl.float32)

    for k_start in range(0, IC_KHKW, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)            # [BLOCK_K]
        k_mask = k_offs < IC_KHKW

        ic_idx = k_offs // (KH * KW)
        rem = k_offs % (KH * KW)
        kh_idx = rem // KW
        kw_idx = rem % KW

        # weight [BLOCK_M, BLOCK_K]
        w_ptrs = w_ptr + (oc_offs[:, None] * (IC * KH * KW)
                          + ic_idx[None, :] * (KH * KW)
                          + kh_idx[None, :] * KW
                          + kw_idx[None, :])
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # input [BLOCK_K, NCOLS]
        ih = oh_flat[None, :] + kh_idx[:, None]   # [BLOCK_K, NCOLS]
        iw = ow_flat[None, :] + kw_idx[:, None]   # [BLOCK_K, NCOLS]
        x_ptrs = x_ptr + (pid_n * (IC * H * W)
                          + ic_idx[:, None] * (H * W)
                          + ih * W
                          + iw)
        x_mask = (k_mask[:, None] & valid_col[None, :]
                  & (ih < H) & (iw < W) & (ih >= 0) & (iw >= 0))
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        acc += tl.dot(w_vals, x_vals)

    # bias + subtract
    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)   # [BLOCK_M]
    acc = acc + bias[:, None] - SUBV

    # hardswish
    shifted = tl.minimum(tl.maximum(acc + 3.0, 0.0), 6.0)
    hs = acc * shifted * (1.0 / 6.0)

    # mask invalid columns to -inf for the max
    NEG_INF = float('-inf')
    hs = tl.where(valid_col[None, :], hs, NEG_INF)

    # reshape to (BLOCK_M, BLOCK_N, POOL_AREA) and max-reduce along POOL_AREA
    hs3 = tl.reshape(hs, (BLOCK_M, BLOCK_N, POOL_AREA))
    pooled = tl.max(hs3, axis=2)   # [BLOCK_M, BLOCK_N]

    # mish: x * tanh(softplus(x))
    sp_val = tl.log(1.0 + tl.exp(pooled))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = pooled * tanh_sp

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
        POOL_AREA = POOL_K * POOL_K

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(POOLED_HW, meta['BLOCK_N']),
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
            POOL_AREA,
            IC_KHKW,
        )

        return out