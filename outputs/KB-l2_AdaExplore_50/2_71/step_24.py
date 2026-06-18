import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_SP': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_SP': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_SP': 128}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'OH', 'OW', 'KH', 'KW'],
)
@triton.jit
def conv2d_fused_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    INV_DIV: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)

    oc_mask = oc_offs < OC
    sp_mask = sp_offs < (OH * OW)

    oh = sp_offs // OW
    ow = sp_offs % OW
    x_base = pid_n * (IC * IH * IW)

    K = IC * KH * KW

    acc = tl.zeros((BLOCK_OC, BLOCK_SP), dtype=tl.float32)

    # Use tl.dot via [BLOCK_OC, K_chunk] x [K_chunk, BLOCK_SP] tile
    # Build a small chunk: process one (kh, kw) at a time, accumulate over IC via dot
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            # weights: [BLOCK_OC, IC] for this (kh, kw)
            ic_range = tl.arange(0, IC)
            w_off = oc_offs[:, None] * K + ic_range[None, :] * (KH * KW) + kh * KW + kw
            w_mask = oc_mask[:, None]
            w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)  # [BLOCK_OC, IC]

            # x: [IC, BLOCK_SP] for this (kh, kw)
            x_off = x_base + ic_range[:, None] * (IH * IW) + (oh + kh)[None, :] * IW + (ow + kw)[None, :]
            x_mask = sp_mask[None, :]
            x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [IC, BLOCK_SP]

            acc += tl.dot(w_tile, x_tile, allow_tf32=True)

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[:, None]
    acc = acc * INV_DIV
    acc = tl.where(acc >= 0, acc, acc * NEG_SLOPE)

    out_off = pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + sp_offs[None, :]
    mask = oc_mask[:, None] & sp_mask[None, :]
    tl.store(out_ptr + out_off, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
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

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(OH * OW, META['BLOCK_SP']))

        conv2d_fused_kernel[grid](
            x, w, b, out,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            1.0 / float(self.divisor),
            0.01,
        )
        return out