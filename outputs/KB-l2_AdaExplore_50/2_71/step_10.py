import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 16, 'BLOCK_OW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OH': 8, 'BLOCK_OW': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_OH': 2, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OH': 4, 'BLOCK_OW': 128}, num_warps=8, num_stages=3),
    ],
    key=['OH', 'OW', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_div_lrelu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, H, W,
    OC,
    OH, OW,
    inv_div,
    neg_slope,
    KH: tl.constexpr,
    KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_OH: tl.constexpr,
    BLOCK_OW: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_sp = tl.program_id(1)
    num_ow_tiles = tl.cdiv(OW, BLOCK_OW)
    pid_oh = pid_sp // num_ow_tiles
    pid_ow = pid_sp % num_ow_tiles

    oh_offs = pid_oh * BLOCK_OH + tl.arange(0, BLOCK_OH)  # [BLOCK_OH]
    ow_offs = pid_ow * BLOCK_OW + tl.arange(0, BLOCK_OW)  # [BLOCK_OW]
    oc_offs = tl.arange(0, BLOCK_OC)                      # [BLOCK_OC]

    mask_oh = oh_offs < OH
    mask_ow = ow_offs < OW

    sp_mask_2d = mask_oh[:, None] & mask_ow[None, :]  # [BLOCK_OH, BLOCK_OW]

    # acc: [BLOCK_OH * BLOCK_OW, BLOCK_OC]
    acc = tl.zeros((BLOCK_OH * BLOCK_OW, BLOCK_OC), dtype=tl.float32)

    HW = H * W
    KHW = KH * KW

    # base x pointer for this b
    x_b_base = pid_b * (IC * HW)

    for ic in range(0, IC):
        x_ic_base = x_b_base + ic * HW
        w_ic_base = oc_offs * (IC * KHW) + ic * KHW
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh_offs + kh  # [BLOCK_OH]
                iw = ow_offs + kw  # [BLOCK_OW]
                x_off = x_ic_base + ih[:, None] * W + iw[None, :]
                x_vals = tl.load(x_ptr + x_off, mask=sp_mask_2d, other=0.0)  # [BLOCK_OH, BLOCK_OW]
                x_flat = tl.reshape(x_vals, (BLOCK_OH * BLOCK_OW,))

                w_off = w_ic_base + kh * KW + kw  # [BLOCK_OC]
                w_vals = tl.load(w_ptr + w_off)

                acc += x_flat[:, None] * w_vals[None, :]

    bias = tl.load(b_ptr + oc_offs)
    acc = acc + bias[None, :]
    acc = acc * inv_div
    acc = tl.where(acc >= 0, acc, acc * neg_slope)

    # store: out[b, oc, oh, ow], layout [B, OC, OH, OW]
    sp_idx_2d = oh_offs[:, None] * OW + ow_offs[None, :]  # [BLOCK_OH, BLOCK_OW]
    sp_idx = tl.reshape(sp_idx_2d, (BLOCK_OH * BLOCK_OW,))
    sp_mask = tl.reshape(sp_mask_2d, (BLOCK_OH * BLOCK_OW,))

    out_off = pid_b * (OC * OH * OW) + oc_offs[None, :] * (OH * OW) + sp_idx[:, None]
    store_mask = sp_mask[:, None]
    tl.store(out_ptr + out_off, acc, mask=store_mask)


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
            triton.cdiv(OH, meta['BLOCK_OH']) * triton.cdiv(OW, meta['BLOCK_OW']),
        )

        conv2d_div_lrelu_kernel[grid](
            x, weight, bias, out,
            B, IC, H, W,
            OC,
            OH, OW,
            inv_div,
            neg_slope,
            KH=KH, KW=KW,
            BLOCK_OC=OC,
        )
        return out