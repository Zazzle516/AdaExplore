import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KHKW'],
)
@triton.jit
def conv_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,           # conv output spatial
    POH, POW,         # pooled output spatial
    POOL_K: tl.constexpr,
    SUBV,
    OUT_HW,           # POH*POW
    IC_KHKW,          # IC*KH*KW
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program over (N_idx, OC_tile, Pool_spatial_tile)
    pid_n = tl.program_id(0)       # batch index
    pid_oc = tl.program_id(1)      # OC tile (BLOCK_M along OC)
    pid_sp = tl.program_id(2)      # pooled-spatial tile (BLOCK_N along POH*POW)

    oc_offs = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    sp_offs = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N] indices into pooled spatial

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < OUT_HW

    # pooled spatial coords
    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # We'll iterate over pool window (POOL_K x POOL_K) positions, computing conv output at each,
    # apply x - subv -> hardswish, take max over the window, then mish at the end.

    # Initialize max accumulator to -inf
    NEG_INF = float('-inf')
    max_acc = tl.full((BLOCK_M, BLOCK_N), NEG_INF, dtype=tl.float32)

    for pi in tl.static_range(0, POOL_K):
        for pj in tl.static_range(0, POOL_K):
            # conv output coords
            oh = poh * POOL_K + pi    # [BLOCK_N]
            ow = pow_ * POOL_K + pj   # [BLOCK_N]
            valid_sp = (oh < OH) & (ow < OW) & sp_mask

            # accumulator for conv at these (oc, sp)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # GEMM-like reduction over IC*KH*KW
            for k_start in range(0, IC_KHKW, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)         # [BLOCK_K]
                k_mask = k_offs < IC_KHKW

                # decompose k into ic, kh, kw
                ic_idx = k_offs // (KH * KW)
                rem = k_offs % (KH * KW)
                kh_idx = rem // KW
                kw_idx = rem % KW

                # weight: w[oc, ic, kh, kw], shape (OC, IC, KH, KW)
                w_ptrs = w_ptr + (oc_offs[:, None] * (IC * KH * KW)
                                  + ic_idx[None, :] * (KH * KW)
                                  + kh_idx[None, :] * KW
                                  + kw_idx[None, :])
                w_mask = oc_mask[:, None] & k_mask[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)   # [BLOCK_M, BLOCK_K]

                # input coords
                ih = oh[:, None] + kh_idx[None, :]   # [BLOCK_N, BLOCK_K]
                iw = ow[:, None] + kw_idx[None, :]   # [BLOCK_N, BLOCK_K]
                # n is pid_n
                x_ptrs = x_ptr + (pid_n * (IC * H * W)
                                  + ic_idx[None, :] * (H * W)
                                  + ih * W
                                  + iw)
                x_mask = valid_sp[:, None] & k_mask[None, :] & (ih < H) & (iw < W)
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)   # [BLOCK_N, BLOCK_K]

                # acc[BLOCK_M, BLOCK_N] += w[BLOCK_M, BLOCK_K] @ x[BLOCK_N, BLOCK_K]^T
                acc += tl.dot(w_vals, tl.trans(x_vals))

            # add bias
            bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)   # [BLOCK_M]
            acc = acc + bias[:, None]

            # subtract value
            acc = acc - SUBV

            # hardswish: x * relu6(x+3)/6
            shifted = acc + 3.0
            shifted = tl.minimum(tl.maximum(shifted, 0.0), 6.0)
            hs = acc * shifted * (1.0 / 6.0)

            # mask out invalid positions with -inf so they don't affect max
            hs = tl.where(valid_sp[None, :], hs, NEG_INF)

            max_acc = tl.maximum(max_acc, hs)

    # Apply mish: x * tanh(softplus(x))  where softplus(x) = log(1+exp(x))
    # tanh(y) = 1 - 2/(exp(2y)+1)
    sp_val = tl.log(1.0 + tl.exp(max_acc))
    e2 = tl.exp(2.0 * sp_val)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    out = max_acc * tanh_sp

    # store: out[pid_n, oc_offs, sp_offs]
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

        conv_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            POH, POW,
            POOL_K,
            self.subtract_value,
            OUT_HW,
            IC_KHKW,
        )

        return out.view(N, OC, POH, POW)