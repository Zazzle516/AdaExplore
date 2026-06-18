import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OW': 16, 'BLOCK_OC': 32, 'BLOCK_OH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 16, 'BLOCK_OC': 32, 'BLOCK_OH': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OC': 32, 'BLOCK_OH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OC': 32, 'BLOCK_OH': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_OH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OC': 64, 'BLOCK_OH': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OC': 32, 'BLOCK_OH': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OC': 32, 'BLOCK_OH': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 64, 'BLOCK_OC': 64, 'BLOCK_OH': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_OH': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 16, 'BLOCK_OC': 64, 'BLOCK_OH': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OW': 32, 'BLOCK_OC': 32, 'BLOCK_OH': 8}, num_warps=4, num_stages=2),
    ],
    key=['OH', 'OW', 'OC', 'IC'],
)
@triton.jit
def conv2d_direct_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC: tl.constexpr, H, W,
    OC, OH, OW,
    inv_div, neg_slope,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_sp = tl.program_id(2)

    n_ow_tiles = tl.cdiv(OW, BLOCK_OW)
    n_oh_tiles = tl.cdiv(OH, BLOCK_OH)
    pid_oh = pid_sp // n_ow_tiles
    pid_ow = pid_sp % n_ow_tiles

    offs_ow = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    offs_oh = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)  # [BLOCK_OC]

    mask_ow = offs_ow < OW
    mask_oh = offs_oh < OH
    mask_oc = offs_oc < OC

    # 3D accumulator: [BLOCK_OH, BLOCK_OW, BLOCK_OC]
    acc = tl.zeros((BLOCK_OH, BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    HW = H * W
    x_b_base = pid_b * IC * HW

    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                # input position: ih = oh + kh, iw = ow + kw
                ih = offs_oh + kh  # [BLOCK_OH]
                iw = offs_ow + kw  # [BLOCK_OW]
                x_off = x_b_base + ic * HW + ih[:, None] * W + iw[None, :]  # [BLOCK_OH, BLOCK_OW]
                x_mask = (ih[:, None] < H) & (iw[None, :] < W)
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)  # [BLOCK_OH, BLOCK_OW]

                # weight: [OC, IC, KH, KW]
                w_off = offs_oc * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw  # [BLOCK_OC]
                w_val = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]

                acc += x_val[:, :, None] * w_val[None, None, :]

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, None, :]
    acc = acc * inv_div
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store: out[b, oc, oh, ow]
    out_b_base = pid_b * OC * OH * OW
    out_off = (out_b_base
               + offs_oc[None, None, :] * (OH * OW)
               + offs_oh[:, None, None] * OW
               + offs_ow[None, :, None])
    out_mask = (mask_oh[:, None, None] & mask_ow[None, :, None] & mask_oc[None, None, :])
    tl.store(out_ptr + out_off, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = float(divisor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        weight = self.conv.weight.contiguous().cuda()
        bias = self.conv.bias.contiguous().cuda()

        B, IC, H, W = x.shape
        OC, _, KH, KW = weight.shape
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)
        inv_div = 1.0 / self.divisor
        neg_slope = 0.01

        grid = lambda meta: (
            B,
            triton.cdiv(OC, meta['BLOCK_OC']),
            triton.cdiv(OH, meta['BLOCK_OH']) * triton.cdiv(OW, meta['BLOCK_OW']),
        )

        conv2d_direct_kernel[grid](
            x, weight, bias, out,
            B, IC, H, W,
            OC, OH, OW,
            inv_div, neg_slope,
            KH=KH, KW=KW,
        )
        return out