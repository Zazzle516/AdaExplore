import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'OD'],
)
@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr, OC_PAD: tl.constexpr,
    OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, ceil(OH*OW / BLOCK_HW))
    n = tl.program_id(0)
    hw_block = tl.program_id(1)

    offs = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    OHW = OH * OW
    mask_hw = offs < OHW
    oh = offs // OW
    ow = offs % OW

    oc_range = tl.arange(0, OC_PAD)
    oc_mask = oc_range < OC

    # Load bias [OC_PAD]
    bias = tl.load(b_ptr + oc_range, mask=oc_mask, other=0.0)

    HW = H * W
    x_n_base = n * IC * D * HW
    base_hw = oh * W + ow  # [BLOCK_HW]

    # Min accumulator over OD: [OC_PAD, BLOCK_HW]
    min_acc = tl.full([OC_PAD, BLOCK_HW], float('inf'), dtype=tl.float32)

    # Iterate over OD on the outside; accumulate conv into [OC_PAD, BLOCK_HW]
    for od in range(0, OD):
        acc = tl.zeros([OC_PAD, BLOCK_HW], dtype=tl.float32)
        for ic in range(0, IC):
            x_ic_base = x_n_base + ic * D * HW
            w_ic_base = ic * KD * KH * KW  # within an oc; full = oc*IC*KD*KH*KW + ic*KD*KH*KW
            for kd in range(0, KD):
                x_kd_base = x_ic_base + (od + kd) * HW
                w_kd_base = w_ic_base + kd * KH * KW
                for kh in range(0, KH):
                    x_kh_base = x_kd_base + kh * W
                    w_kh_base = w_kd_base + kh * KW
                    for kw in range(0, KW):
                        # x at (n, ic, od+kd, oh+kh, ow+kw): [BLOCK_HW]
                        x_offs = x_kh_base + kw + base_hw
                        x_val = tl.load(x_ptr + x_offs, mask=mask_hw, other=0.0)
                        # weights for all oc at (ic, kd, kh, kw): [OC_PAD]
                        w_offs = oc_range * (IC * KD * KH * KW) + w_kh_base + kw
                        w_val = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)
                        acc += w_val[:, None] * x_val[None, :]
        acc = acc + bias[:, None]
        min_acc = tl.minimum(min_acc, acc)

    # Mask invalid oc rows (set to -inf so they don't affect softmax max)
    min_acc = tl.where(oc_mask[:, None], min_acc, -float('inf'))

    # Softmax along channel (axis=0)
    m = tl.max(min_acc, axis=0)  # [BLOCK_HW]
    e = tl.exp(min_acc - m[None, :])
    e = tl.where(oc_mask[:, None], e, 0.0)
    z = tl.sum(e, axis=0)  # [BLOCK_HW]
    y = e / z[None, :]

    # Store: out shape (N, OC, OH*OW)
    out_offs = n * OC * OHW + oc_range[:, None] * OHW + offs[None, :]
    store_mask = oc_mask[:, None] & mask_hw[None, :]
    tl.store(out_ptr + out_offs, y, mask=store_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        if self.dim != 2:
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            return torch.softmax(y, dim=1)

        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, D, H, W = x.shape
        OC = self.out_channels
        KD = KH = KW = self.kernel_size
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        OC_PAD = _next_pow2(OC)

        grid = lambda META: (N, (OH * OW + META['BLOCK_HW'] - 1) // META['BLOCK_HW'])
        conv3d_min_softmax_kernel[grid](
            x, w, b, out,
            N, IC, D, H, W,
            OC, OC_PAD, OD, OH, OW,
            KD, KH, KW,
        )
        return out