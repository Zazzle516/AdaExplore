import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_S': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_S': 256}, num_warps=4, num_stages=2),
    ],
    key=['IC', 'OC', 'H', 'W', 'KH', 'KW'],
)
@triton.jit
def conv_gelu_avgpool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, OC: tl.constexpr,
    H: tl.constexpr, W: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    OHOW = OH * OW
    HW = H * W
    ICHW = IC * HW
    KHKW = KH * KW
    ICKHKW = IC * KHKW
    inv_area = 1.0 / (OH * OW)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # Preload weights for this OC tile: shape [BLOCK_OC, IC*KH*KW]
    # We'll just reload inside the loop, but hoist scalar constants.

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)

    # Accumulator over all spatial positions for this (n, oc tile)
    out_acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)

    x_base = pid_n * ICHW

    num_s_tiles = (OHOW + BLOCK_S - 1) // BLOCK_S
    for s_tile in range(0, num_s_tiles):
        s_offs = s_tile * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < OHOW

        oh = s_offs // OW
        ow = s_offs % OW

        acc = tl.zeros((BLOCK_S, BLOCK_OC), dtype=tl.float32)

        for ic in tl.static_range(0, IC):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    ih = oh + kh
                    iw = ow + kw
                    x_idx = x_base + ic * HW + ih * W + iw
                    x_val = tl.load(x_ptr + x_idx, mask=s_mask, other=0.0)
                    w_idx = oc_offs * ICKHKW + ic * KHKW + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)
                    acc += x_val[:, None] * w_val[None, :]

        acc = acc + bias[None, :]
        inv_sqrt2 = 0.70710678118654752440
        gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
        gelu = tl.where(s_mask[:, None], gelu, 0.0)
        out_acc += tl.sum(gelu, axis=0)

    scaled = out_acc * inv_area
    out_offs = pid_n * OC + oc_offs
    tl.store(out_ptr + out_offs, scaled, mask=oc_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.cuda().contiguous()
        weight = self.conv.weight.cuda().contiguous()
        bias = self.conv.bias.cuda().contiguous()

        N, IC, H, W = x.shape
        OC = self.out_channels
        KH = KW = self.kernel_size
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC), device=x.device, dtype=torch.float32)

        grid = lambda META: (N, (OC + META['BLOCK_OC'] - 1) // META['BLOCK_OC'])

        conv_gelu_avgpool_kernel[grid](
            x, weight, bias, out,
            N, IC, OC, H, W, OH, OW, KH, KW,
        )

        return out