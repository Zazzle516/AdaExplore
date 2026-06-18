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
    ],
    key=['OC', 'OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,          # conv output spatial
    POH, POW,        # pooled spatial
    POOL_K,
    SUBV,
    OC_stride_oc, OC_stride_ic, OC_stride_kh, OC_stride_kw,
    OUT_HW: tl.constexpr,
    IC_KH_KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n_oc = tl.program_id(0)  # combined batch * oc_tile
    pid_sp = tl.program_id(1)    # spatial tile (over pooled output)

    num_oc_tiles = tl.cdiv(OC, BLOCK_N)
    n_idx = pid_n_oc // num_oc_tiles
    oc_tile = pid_n_oc % num_oc_tiles

    # spatial index in pooled output, BLOCK_M positions
    sp_start = pid_sp * BLOCK_M
    sp_offs = sp_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    # For each pooled position, we need to compute conv output at POOL_K x POOL_K positions
    # then max-pool, then mish.
    # We accumulate conv at (POOL_K*POOL_K) positions per pooled output.
    # Approach: for each of the POOL_K*POOL_K sub-positions, compute the GEMM tile and combine.

    oc_offs = oc_tile * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    oc_mask = oc_offs < OC

    neg_inf = float('-inf')
    max_acc = tl.full((BLOCK_M, BLOCK_N), neg_inf, dtype=tl.float32)

    for ph in tl.static_range(0, 16):
        if ph < POOL_K:
            for pw in tl.static_range(0, 16):
                if pw < POOL_K:
                    # conv output position
                    oh = poh * POOL_K + ph
                    ow = pow_ * POOL_K + pw
                    valid = sp_mask & (oh < OH) & (ow < OW)

                    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

                    # GEMM over IC*KH*KW
                    for k_start in range(0, IC_KH_KW, BLOCK_K):
                        k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                        k_mask = k_offs < IC_KH_KW

                        ic = k_offs // (KH * KW)
                        rem = k_offs % (KH * KW)
                        kh = rem // KW
                        kw = rem % KW

                        # input H, W positions
                        ih = oh[:, None] + kh[None, :]  # [BLOCK_M, BLOCK_K]
                        iw = ow[:, None] + kw[None, :]

                        x_off = ((n_idx * IC + ic[None, :]) * H + ih) * W + iw  # [BLOCK_M, BLOCK_K]
                        x_mask = valid[:, None] & k_mask[None, :]
                        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                        # weight
                        w_off = (oc_offs[:, None] * OC_stride_oc
                                 + ic[None, :] * OC_stride_ic
                                 + kh[None, :] * OC_stride_kh
                                 + kw[None, :] * OC_stride_kw)  # [BLOCK_N, BLOCK_K]
                        w_mask = oc_mask[:, None] & k_mask[None, :]
                        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                        acc += tl.dot(x_vals, tl.trans(w_vals))

                    # add bias
                    bvals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
                    acc = acc + bvals[None, :]

                    # subtract
                    acc = acc - SUBV

                    # hardswish: x * relu6(x+3)/6
                    t = acc + 3.0
                    t = tl.maximum(t, 0.0)
                    t = tl.minimum(t, 6.0)
                    hs = acc * t * (1.0 / 6.0)

                    # mask invalid positions to -inf
                    hs = tl.where(valid[:, None], hs, neg_inf)

                    max_acc = tl.maximum(max_acc, hs)

    # Apply mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)); use stable form: max(x,0) + log(1+exp(-|x|))
    x = max_acc
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = x * th

    # store
    out_off = ((n_idx * OC + oc_offs[None, :]) * POH * POW) + sp_offs[:, None]
    out_mask = sp_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_off, out, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = int(pool_kernel_size)
        self.kernel_size = int(kernel_size)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

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

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        IC_KH_KW = IC * KH * KW
        OUT_HW = POH * POW

        sw = w.stride()

        def grid(meta):
            num_oc_tiles = triton.cdiv(OC, meta['BLOCK_N'])
            num_sp_tiles = triton.cdiv(OUT_HW, meta['BLOCK_M'])
            return (N * num_oc_tiles, num_sp_tiles)

        conv_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            POH, POW,
            POOL_K,
            self.subtract_value,
            sw[0], sw[1], sw[2], sw[3],
            OUT_HW=OUT_HW,
            IC_KH_KW=IC_KH_KW,
        )
        return out