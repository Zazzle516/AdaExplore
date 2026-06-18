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
    SUBV,
    OC_stride_oc, OC_stride_ic, OC_stride_kh, OC_stride_kw,
    POOL_K: tl.constexpr,
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

    # spatial index in pooled output
    sp_start = pid_sp * BLOCK_M
    sp_offs = sp_start + tl.arange(0, BLOCK_M)
    sp_mask = sp_offs < (POH * POW)

    poh = sp_offs // POW
    pow_ = sp_offs % POW

    oc_offs = oc_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    oc_mask = oc_offs < OC

    # POOL_K*POOL_K accumulators
    acc00 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # GEMM over IC*KH*KW, hoisting weight load above the ph,pw loops
    for k_start in range(0, IC_KH_KW, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < IC_KH_KW

        ic = k_offs // (KH * KW)
        rem = k_offs % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # weight load - shared across all pool positions
        w_off = (oc_offs[:, None] * OC_stride_oc
                 + ic[None, :] * OC_stride_ic
                 + kh[None, :] * OC_stride_kh
                 + kw[None, :] * OC_stride_kw)
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
        w_t = tl.trans(w_vals)

        # ph=0,pw=0
        oh0 = poh * POOL_K + 0
        ow0 = pow_ * POOL_K + 0
        ih = oh0[:, None] + kh[None, :]
        iw = ow0[:, None] + kw[None, :]
        x_off = ((n_idx * IC + ic[None, :]) * H + ih) * W + iw
        x_mask = sp_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
        acc00 += tl.dot(x_vals, w_t)

        # ph=0,pw=1
        ow1 = pow_ * POOL_K + 1
        iw = ow1[:, None] + kw[None, :]
        x_off = ((n_idx * IC + ic[None, :]) * H + ih) * W + iw
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
        acc01 += tl.dot(x_vals, w_t)

        # ph=1,pw=0
        oh1 = poh * POOL_K + 1
        ih = oh1[:, None] + kh[None, :]
        iw = ow0[:, None] + kw[None, :]
        x_off = ((n_idx * IC + ic[None, :]) * H + ih) * W + iw
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
        acc10 += tl.dot(x_vals, w_t)

        # ph=1,pw=1
        iw = ow1[:, None] + kw[None, :]
        x_off = ((n_idx * IC + ic[None, :]) * H + ih) * W + iw
        x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
        acc11 += tl.dot(x_vals, w_t)

    # add bias
    bvals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    bv = bvals[None, :]

    def _hs(a):
        a = a + bv - SUBV
        t = a + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        return a * t * (1.0 / 6.0)

    hs00 = _hs(acc00)
    hs01 = _hs(acc01)
    hs10 = _hs(acc10)
    hs11 = _hs(acc11)

    m = tl.maximum(tl.maximum(hs00, hs01), tl.maximum(hs10, hs11))

    # mish
    abs_x = tl.abs(m)
    sp = tl.maximum(m, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = m * th

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
            self.subtract_value,
            sw[0], sw[1], sw[2], sw[3],
            POOL_K=POOL_K,
            OUT_HW=OUT_HW,
            IC_KH_KW=IC_KH_KW,
        )
        return out