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
    # program_id(0): n
    # program_id(1): oc tile
    # program_id(2): pooled-spatial tile (POH*POW pooled positions, each containing POOL*POOL conv outputs)
    n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)  # [BM]
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)  # [BN] indexes pooled positions

    # pooled position -> (poh, pow)
    poh = offs_sp // POW
    pow_ = offs_sp % POW

    P2 = POOL * POOL

    # accumulator for max over pool window: [BM, BN]
    neg_inf = float('-inf')
    max_acc = tl.full((BLOCK_M, BLOCK_N), neg_inf, dtype=tl.float32)

    # Loop over pool positions
    for pp in range(0, P2):
        ph_off = pp // POOL
        pw_off = pp % POOL
        # conv output coords
        oh = poh * POOL + ph_off  # [BN]
        ow = pow_ * POOL + pw_off  # [BN]

        # Compute conv at (n, oc[BM], oh,ow [BN]) via GEMM along K_TOTAL = IC*KH*KW
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k0 in range(0, K_TOTAL, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)  # [BK]
            k_mask = offs_k < K_TOTAL

            # decode k -> (ic, kh, kw)
            ic = offs_k // (KH * KW)
            rem = offs_k % (KH * KW)
            kh = rem // KW
            kw = rem % KW

            # Weight: w[oc, ic, kh, kw] -> [BM, BK]
            w_offs = (offs_oc[:, None] * (IC * KH * KW) +
                      ic[None, :] * (KH * KW) +
                      kh[None, :] * KW +
                      kw[None, :])
            w_mask = (offs_oc[:, None] < OC) & (k_mask[None, :])
            w = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

            # Input: x[n, ic, oh+kh, ow+kw] -> [BK, BN]
            ih = oh[None, :] + kh[:, None]  # [BK, BN]
            iw = ow[None, :] + kw[:, None]  # [BK, BN]
            x_offs = (n * (IC * IH * IW) +
                      ic[:, None] * (IH * IW) +
                      ih * IW +
                      iw)
            x_mask = (k_mask[:, None] &
                      (offs_sp[None, :] < OUT_SPATIAL) &
                      (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW))
            x = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

            acc += tl.dot(w, x)

        # add bias
        bias = tl.load(b_ptr + offs_oc, mask=offs_oc < OC, other=0.0)
        acc = acc + bias[:, None]

        # subtract value
        acc = acc - SUB

        # hardswish: x * relu6(x+3)/6
        t = acc + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = acc * t * (1.0 / 6.0)

        max_acc = tl.maximum(max_acc, hs)

    # Apply mish: x * tanh(softplus(x))
    # softplus(x) = log(1+exp(x)); use stable form
    x_val = max_acc
    # softplus stable
    sp = tl.where(x_val > 20.0, x_val, tl.log(1.0 + tl.exp(tl.minimum(x_val, 20.0))))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out_val = x_val * th

    # store
    out_offs = (n * (OC * POH * POW) +
                offs_oc[:, None] * (POH * POW) +
                offs_sp[None, :])
    out_mask = (offs_oc[:, None] < OC) & (offs_sp[None, :] < OUT_SPATIAL)
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