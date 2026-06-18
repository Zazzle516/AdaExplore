import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW'],
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

    base_hw = oh * W + ow  # [BLOCK_HW]
    x_n_base = n * IC * D * H * W

    offs_oc = tl.arange(0, OC)  # [OC]
    bias = tl.load(b_ptr + offs_oc)  # [OC]

    # min_val shape: [OC, BLOCK_HW]
    min_val = tl.full([OC, BLOCK_HW], float('inf'), dtype=tl.float32)

    for od in range(0, OD):
        acc = tl.zeros([OC, BLOCK_HW], dtype=tl.float32)
        for ic in range(0, IC):
            x_ic_base = x_n_base + ic * D * H * W
            for kd in range(0, KD):
                id_ = od + kd
                x_d_base = x_ic_base + id_ * H * W
                for kh in range(0, KH):
                    x_kh_base = x_d_base + kh * W + base_hw  # [BLOCK_HW]
                    for kw in range(0, KW):
                        x_val = tl.load(x_ptr + x_kh_base + kw, mask=mask_hw, other=0.0)  # [BLOCK_HW]
                        # weight: (OC, IC, KD, KH, KW)
                        w_off = offs_oc * (IC * KD * KH * KW) + ic * KD * KH * KW + kd * KH * KW + kh * KW + kw
                        w_val = tl.load(w_ptr + w_off)  # [OC]
                        acc += w_val[:, None] * x_val[None, :]
        acc = acc + bias[:, None]
        min_val = tl.minimum(min_val, acc)

    # Now softmax along OC axis for each spatial position
    # mask invalid spatial positions
    min_val = tl.where(mask_hw[None, :], min_val, float('inf'))
    m = tl.max(min_val, axis=0)  # [BLOCK_HW]
    e = tl.exp(min_val - m[None, :])
    z = tl.sum(e, axis=0)  # [BLOCK_HW]
    y = e / z[None, :]

    # Store output: shape (N, OC, OH, OW) -> stride: oc * OHW + offs
    out_base = n * OC * OHW
    out_ptrs = out_ptr + out_base + offs_oc[:, None] * OHW + offs[None, :]
    tl.store(out_ptrs, y, mask=mask_hw[None, :])


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