import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 4, 'BLOCK_PW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 2, 'BLOCK_PW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_PH': 8, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 2, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PH': 8, 'BLOCK_PW': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PH': 2, 'BLOCK_PW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 16, 'BLOCK_PH': 4, 'BLOCK_PW': 8}, num_warps=4, num_stages=3),
    ],
    key=['OC', 'PH', 'PW', 'IC_C'],
)
@triton.jit
def conv_pool_sigmoid_sum_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC,
    OH, OW,
    PH, PW,
    stride_xn, stride_xc, stride_xh,
    KH: tl.constexpr,
    KW: tl.constexpr,
    POOL: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_PH: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_p = tl.program_id(2)

    num_pw_blocks = (PW + BLOCK_PW - 1) // BLOCK_PW
    pid_ph = pid_p // num_pw_blocks
    pid_pw = pid_p % num_pw_blocks

    oc_off = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    ph_off = pid_ph * BLOCK_PH + tl.arange(0, BLOCK_PH)  # [BLOCK_PH]
    pw_off = pid_pw * BLOCK_PW + tl.arange(0, BLOCK_PW)  # [BLOCK_PW]

    oc_mask = oc_off < OC
    ph_mask = ph_off < PH
    pw_mask = pw_off < PW

    # Conv output tile shape: [BLOCK_PH*POOL, BLOCK_PW*POOL]
    OH_TILE: tl.constexpr = BLOCK_PH * POOL
    OW_TILE: tl.constexpr = BLOCK_PW * POOL

    oh_local = tl.arange(0, OH_TILE)  # [OH_TILE]
    ow_local = tl.arange(0, OW_TILE)  # [OW_TILE]

    oh_abs = pid_ph * OH_TILE + oh_local  # conv output row indices
    ow_abs = pid_pw * OW_TILE + ow_local  # conv output col indices

    oh_valid = oh_abs < OH
    ow_valid = ow_abs < OW

    # Accumulator for conv output: [BLOCK_OC, OH_TILE, OW_TILE]
    acc = tl.zeros((BLOCK_OC, OH_TILE, OW_TILE), dtype=tl.float32)

    x_n_base = pid_n * stride_xn

    for ic in tl.static_range(0, IC_C):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_abs[:, None] + kh  # [OH_TILE, 1]... but we need shape [OH_TILE, OW_TILE]
                iw = ow_abs[None, :] + kw
                ih_b = oh_abs + kh  # [OH_TILE]
                iw_b = ow_abs + kw  # [OW_TILE]

                in_mask = oh_valid[:, None] & ow_valid[None, :]
                in_idx = x_n_base + ic * stride_xc + ih_b[:, None] * stride_xh + iw_b[None, :]
                x_val = tl.load(x_ptr + in_idx, mask=in_mask, other=0.0)  # [OH_TILE, OW_TILE]

                w_idx = oc_off * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]

                acc += w_val[:, None, None] * x_val[None, :, :]

    # Add bias
    bias = tl.load(b_ptr + oc_off, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None, None]

    # Now pool: sum POOL×POOL groups
    # Reshape acc [BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL]
    acc_r = tl.reshape(acc, (BLOCK_OC, BLOCK_PH, POOL, BLOCK_PW, POOL))
    pooled = tl.sum(tl.sum(acc_r, axis=4), axis=2)  # [BLOCK_OC, BLOCK_PH, BLOCK_PW]
    pooled = pooled / (POOL * POOL)

    sig = tl.sigmoid(pooled)

    full_mask = oc_mask[:, None, None] & ph_mask[None, :, None] & pw_mask[None, None, :]
    sig = tl.where(full_mask, sig, 0.0)

    s = tl.sum(tl.sum(tl.sum(sig, axis=2), axis=1), axis=0)
    tl.atomic_add(out_ptr + pid_n, s)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        POOL = self.pool_kernel_size

        OH = H - KH + 1
        OW = W - KW + 1
        PH = OH // POOL
        PW = OW // POOL

        out = torch.zeros(N, device=x.device, dtype=torch.float32)

        grid = lambda META: (
            N,
            (OC + META['BLOCK_OC'] - 1) // META['BLOCK_OC'],
            ((PH + META['BLOCK_PH'] - 1) // META['BLOCK_PH']) *
            ((PW + META['BLOCK_PW'] - 1) // META['BLOCK_PW']),
        )

        conv_pool_sigmoid_sum_kernel[grid](
            x, w, b, out,
            N, IC, H, W,
            OC,
            OH, OW,
            PH, PW,
            x.stride(0), x.stride(1), x.stride(2),
            KH=KH,
            KW=KW,
            POOL=POOL,
            IC_C=IC,
        )

        return out