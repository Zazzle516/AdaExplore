import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'OD'],
)
@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H: tl.constexpr, W: tl.constexpr,
    OC: tl.constexpr, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
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

    HW = H * W
    x_n_base = n * IC * D * HW
    base_hw = oh * W + ow  # [BLOCK_HW]

    oc_range = tl.arange(0, OC)  # [OC]

    # Initialize min accumulator across od with +inf
    min_acc = tl.full([OC, BLOCK_HW], float('inf'), dtype=tl.float32)

    # Loop over od first; inside compute conv at this od for all (OC, BLOCK_HW)
    for od in range(0, OD):
        acc = tl.zeros([OC, BLOCK_HW], dtype=tl.float32)
        for ic in range(0, IC):
            x_ic_base = x_n_base + ic * D * HW
            for kd in range(0, KD):
                x_d_base = x_ic_base + (od + kd) * HW
                for kh in range(0, KH):
                    x_kh_base = x_d_base + kh * W
                    for kw in range(0, KW):
                        # x: [BLOCK_HW]
                        x_offs = x_kh_base + kw + base_hw
                        x_val = tl.load(x_ptr + x_offs, mask=mask_hw, other=0.0)
                        # w: [OC]  -- w[oc, ic, kd, kh, kw]
                        w_off = ((oc_range * IC + ic) * KD + kd) * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off)
                        acc += w_val[:, None] * x_val[None, :]
        # Add bias
        bias = tl.load(b_ptr + oc_range)
        acc = acc + bias[:, None]
        min_acc = tl.minimum(min_acc, acc)

    # Now do softmax along OC axis for each hw
    m = tl.max(min_acc, axis=0)  # [BLOCK_HW]
    e = tl.exp(min_acc - m[None, :])
    z = tl.sum(e, axis=0)  # [BLOCK_HW]
    y = e / z[None, :]

    # Store: out shape (N, OC, OH, OW), contiguous
    out_off = (n * OC + oc_range[:, None]) * OHW + offs[None, :]
    store_mask = mask_hw[None, :]
    tl.store(out_ptr + out_off, y, mask=store_mask)


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

        grid = lambda META: (N, (OH * OW + META['BLOCK_HW'] - 1) // META['BLOCK_HW'])
        conv3d_min_softmax_kernel[grid](
            x, w, b, out,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            KD, KH, KW,
        )
        return out