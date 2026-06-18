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
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC_KH_KW'],
)
@triton.jit
def conv_sub_hswish_pool_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, KH, KW,
    OH, OW,         # conv output H,W
    POH, POW,       # pooled output H,W
    POOL_K,
    SUB_VAL,
    OUT_HW,         # POH*POW
    IC_KH_KW,       # IC*KH*KW
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # output spatial tile (pooled positions)
    BLOCK_K: tl.constexpr,  # K reduction tile
):
    pid_n = tl.program_id(0)        # batch
    pid_oc = tl.program_id(1)       # OC tile
    pid_sp = tl.program_id(2)       # pooled spatial tile

    offs_oc = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    offs_sp = pid_sp * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N] pooled flat indices

    mask_oc = offs_oc < OC
    mask_sp = offs_sp < OUT_HW

    # pooled output positions
    poh = offs_sp // POW
    pow_ = offs_sp % POW

    # We need to compute pooling: max over POOL_K x POOL_K of conv outputs.
    # Conv output positions for pooled cell (poh,pow):
    #   oh in [poh*POOL_K, poh*POOL_K + POOL_K - 1], same for ow.

    # Init max accumulator
    NEG_INF = float('-inf')
    max_acc = tl.full((BLOCK_M, BLOCK_N), NEG_INF, dtype=tl.float32)

    # Loop over pool window
    for ph in tl.static_range(0, POOL_K):
        for pw in tl.static_range(0, POOL_K):
            oh = poh * POOL_K + ph    # [BLOCK_N]
            ow = pow_ * POOL_K + pw   # [BLOCK_N]

            # GEMM-style accumulation for these output positions and OC tile
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for k_start in range(0, IC_KH_KW, BLOCK_K):
                k_offs = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                k_mask = k_offs < IC_KH_KW

                # decode k -> ic, kh, kw
                ic = k_offs // (KH * KW)
                rem = k_offs % (KH * KW)
                kh = rem // KW
                kw = rem % KW

                # weight: [OC, IC, KH, KW] -> w[oc, ic, kh, kw]
                w_idx = (offs_oc[:, None] * (IC * KH * KW)
                         + ic[None, :] * (KH * KW)
                         + kh[None, :] * KW
                         + kw[None, :])
                w_mask = mask_oc[:, None] & k_mask[None, :]
                w_vals = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

                # input position
                ih = oh[:, None] + kh[None, :]   # [BLOCK_N, BLOCK_K]
                iw = ow[:, None] + kw[None, :]   # [BLOCK_N, BLOCK_K]

                in_bounds = (oh[:, None] < OH) & (ow[:, None] < OW)
                x_idx = (pid_n * (IC * IH * IW)
                         + ic[None, :] * (IH * IW)
                         + ih * IW
                         + iw)
                x_mask = in_bounds & k_mask[None, :]
                x_vals = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
                # x_vals: [BLOCK_N, BLOCK_K]; w_vals: [BLOCK_M, BLOCK_K]
                acc += tl.dot(w_vals, tl.trans(x_vals))  # [BLOCK_M, BLOCK_N]

            # add bias
            bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
            acc = acc + bias[:, None]

            # subtract
            acc = acc - SUB_VAL

            # hardswish: x * relu6(x+3) / 6
            t = acc + 3.0
            t = tl.minimum(tl.maximum(t, 0.0), 6.0)
            hs = acc * t * (1.0 / 6.0)

            # mask out-of-bounds pool positions to -inf
            valid = (oh[None, :] < OH) & (ow[None, :] < OW)
            hs = tl.where(valid, hs, NEG_INF)

            max_acc = tl.maximum(max_acc, hs)

    # Apply mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)); use stable form
    x = max_acc
    # stable softplus
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(tl.minimum(x, 20.0))))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    out = x * th

    # store
    out_idx = (pid_n * (OC * OUT_HW)
               + offs_oc[:, None] * OUT_HW
               + offs_sp[None, :])
    out_mask = mask_oc[:, None] & mask_sp[None, :]
    tl.store(out_ptr + out_idx, out, mask=out_mask)


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

        N, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1
        PK = self.pool_kernel_size
        POH = OH // PK
        POW = OW // PK

        out = torch.empty((N, OC, POH, POW), device=x.device, dtype=x.dtype)

        OUT_HW = POH * POW
        IC_KH_KW = IC * KH * KW

        grid = lambda meta: (
            N,
            triton.cdiv(OC, meta['BLOCK_M']),
            triton.cdiv(OUT_HW, meta['BLOCK_N']),
        )

        conv_sub_hswish_pool_mish_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, KH, KW,
            OH, OW,
            POH, POW,
            PK,
            self.subtract_value,
            OUT_HW,
            IC_KH_KW,
        )
        return out