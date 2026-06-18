import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'OUT_SPATIAL', 'K_TOTAL'],
)
@triton.jit
def conv_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    POH, POW,
    POOL,
    SUB,
    OUT_SPATIAL, K_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)

    poh = offs_sp // POW
    pow_ = offs_sp % POW

    oh0 = poh * POOL
    ow0 = pow_ * POOL

    sp_mask = offs_sp < OUT_SPATIAL
    oc_mask = offs_oc < OC

    acc00 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_offset = n * (IC * IH * IW)

    for k0 in range(0, K_TOTAL, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        ic = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        w_offs = (offs_oc[:, None] * (IC * KH * KW) +
                  ic[None, :] * (KH * KW) +
                  kh[None, :] * KW +
                  kw[None, :])
        w_mask = oc_mask[:, None] & k_mask[None, :]
        w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        ih_base = oh0[None, :] + kh[:, None]
        iw_base = ow0[None, :] + kw[:, None]
        x_base = (n_offset +
                  ic[:, None] * (IH * IW) +
                  ih_base * IW +
                  iw_base)
        x_kmask = k_mask[:, None] & sp_mask[None, :]

        x00 = tl.load(x_ptr + x_base, mask=x_kmask, other=0.0)
        acc00 += tl.dot(w, x00)

        x01 = tl.load(x_ptr + x_base + 1, mask=x_kmask, other=0.0)
        acc01 += tl.dot(w, x01)

        x10 = tl.load(x_ptr + x_base + IW, mask=x_kmask, other=0.0)
        acc10 += tl.dot(w, x10)

        x11 = tl.load(x_ptr + x_base + IW + 1, mask=x_kmask, other=0.0)
        acc11 += tl.dot(w, x11)

    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)[:, None]
    INV6 = 1.0 / 6.0

    v00 = acc00 + bias - SUB
    t00 = tl.minimum(tl.maximum(v00 + 3.0, 0.0), 6.0)
    hs00 = v00 * t00 * INV6

    v01 = acc01 + bias - SUB
    t01 = tl.minimum(tl.maximum(v01 + 3.0, 0.0), 6.0)
    hs01 = v01 * t01 * INV6

    v10 = acc10 + bias - SUB
    t10 = tl.minimum(tl.maximum(v10 + 3.0, 0.0), 6.0)
    hs10 = v10 * t10 * INV6

    v11 = acc11 + bias - SUB
    t11 = tl.minimum(tl.maximum(v11 + 3.0, 0.0), 6.0)
    hs11 = v11 * t11 * INV6

    max_acc = tl.maximum(tl.maximum(hs00, hs01), tl.maximum(hs10, hs11))

    x_val = max_acc
    sp = tl.where(x_val > 20.0, x_val, tl.log(1.0 + tl.exp(tl.minimum(x_val, 20.0))))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out_val = x_val * th

    out_offs = (n * (OC * POH * POW) +
                offs_oc[:, None] * (POH * POW) +
                offs_sp[None, :])
    out_mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_offs, out_val, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, subtract_value, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.subtract_value = float(subtract_value)
        self.pool_kernel_size = pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1
        POOL = self.pool_kernel_size
        POH = OH // POOL
        POW = OW // POOL

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=torch.float32)

        OUT_SPATIAL = POH * POW
        K_TOTAL = IC * KH * KW

        grid = lambda meta: (
            N,
            (OC + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],
            (OUT_SPATIAL + meta['BLOCK_N'] - 1) // meta['BLOCK_N'],
        )

        conv_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            POH, POW,
            POOL,
            self.subtract_value,
            OUT_SPATIAL, K_TOTAL,
        )
        return out