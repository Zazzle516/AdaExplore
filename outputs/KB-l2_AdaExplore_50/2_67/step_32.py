import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


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
    pid_s = tl.program_id(2)

    OHOW = OH * OW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < OHOW

    oh = s_offs // OW
    ow = s_offs % OW

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    # acc shape: [BLOCK_S, BLOCK_OC]
    acc = tl.zeros((BLOCK_S, BLOCK_OC), dtype=tl.float32)

    x_base = pid_n * IC * H * W

    for ic in tl.static_range(0, IC):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                ih = oh + kh
                iw = ow + kw
                x_idx = x_base + ic * H * W + ih * W + iw
                x_val = tl.load(x_ptr + x_idx, mask=s_mask, other=0.0)  # [BLOCK_S]
                w_idx = oc_offs * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_idx, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                acc += x_val[:, None] * w_val[None, :]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc = acc + bias[None, :]

    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    gelu = tl.where(s_mask[:, None], gelu, 0.0)
    partial_sum = tl.sum(gelu, axis=0)  # [BLOCK_OC]

    scaled = partial_sum / (OH * OW)

    out_offs = pid_n * OC + oc_offs
    tl.atomic_add(out_ptr + out_offs, scaled, mask=oc_mask)


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

        out = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_S = 128
        OHOW = OH * OW
        grid = (N, (OC + BLOCK_OC - 1) // BLOCK_OC, (OHOW + BLOCK_S - 1) // BLOCK_S)

        conv_gelu_avgpool_kernel[grid](
            x, weight, bias, out,
            N, IC, OC, H, W, OH, OW, KH, KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_S=BLOCK_S,
            num_warps=4, num_stages=2,
        )

        return out