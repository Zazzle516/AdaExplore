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
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_HW': 256}, num_warps=4, num_stages=2),
    ],
    key=['N', 'OC', 'OH', 'OW', 'OD'],
)
@triton.jit
def conv3d_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H: tl.constexpr, W: tl.constexpr,
    OC, OD: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OD: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, OC, ceil(OH*OW / BLOCK_HW))
    n = tl.program_id(0)
    oc = tl.program_id(1)
    hw_block = tl.program_id(2)

    offs = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    OHW = OH * OW
    mask_hw = offs < OHW
    oh = offs // OW
    ow = offs % OW

    bias = tl.load(b_ptr + oc)

    base_hw = oh * W + ow  # [BLOCK_HW]
    HW = H * W
    x_n_base = n * IC * D * HW

    od_range = tl.arange(0, BLOCK_OD)
    od_mask = od_range < OD

    # 2D accumulator: [BLOCK_OD, BLOCK_HW]
    acc = tl.zeros([BLOCK_OD, BLOCK_HW], dtype=tl.float32)

    # Loops over (ic, kd, kh, kw) — weights loaded once and reused across all od
    for ic in range(0, IC):
        x_ic_base = x_n_base + ic * D * HW
        w_ic_base = (oc * IC + ic) * KD * KH * KW
        for kd in range(0, KD):
            w_d_base = w_ic_base + kd * KH * KW
            x_kd_base = x_ic_base + kd * HW
            for kh in range(0, KH):
                w_kh_base = w_d_base + kh * KW
                x_kh_base = x_kd_base + kh * W
                for kw in range(0, KW):
                    w_val = tl.load(w_ptr + w_kh_base + kw)
                    # x at (n, ic, od+kd, oh+kh, ow+kw) for od in [0, OD)
                    x_offs = (x_kh_base + kw) + od_range[:, None] * HW + base_hw[None, :]
                    x_mask = od_mask[:, None] & mask_hw[None, :]
                    x_val = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)
                    acc += x_val * w_val

    acc = acc + bias
    # Mask out invalid od slots so they don't affect min
    acc = tl.where(od_mask[:, None], acc, float('inf'))
    min_val = tl.min(acc, axis=0)  # [BLOCK_HW]

    out_off = ((n * OC + oc) * OHW) + offs
    tl.store(out_ptr + out_off, min_val, mask=mask_hw)


@triton.jit
def softmax_channel_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_C: tl.constexpr,
):
    # grid: (N, S)
    n = tl.program_id(0)
    s = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * S + s
    x_ptrs = x_ptr + base + offs_c * S
    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))

    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    e = tl.where(mask_c, e, 0.0)
    z = tl.sum(e, axis=0)
    y = e / z

    out_ptrs = out_ptr + base + offs_c * S
    tl.store(out_ptrs, y, mask=mask_c)


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
            # fallback
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

        out_min = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)

        BLOCK_OD = _next_pow2(OD)
        grid = lambda META: (N, OC, (OH * OW + META['BLOCK_HW'] - 1) // META['BLOCK_HW'])
        conv3d_min_kernel[grid](
            x, w, b, out_min,
            N, IC, D, H, W,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_OD=BLOCK_OD,
        )

        # Softmax along channel dim (dim=1) of (N, OC, OH, OW)
        out = torch.empty_like(out_min)
        S = OH * OW
        BLOCK_C = _next_pow2(OC)
        grid2 = (N, S)
        softmax_channel_kernel[grid2](
            out_min, out,
            N, OC, S,
            BLOCK_C=BLOCK_C,
            num_warps=2,
        )
        return out