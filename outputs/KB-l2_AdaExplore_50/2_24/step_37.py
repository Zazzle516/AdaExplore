import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HW': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 128}, num_warps=8, num_stages=3),
    ],
    key=['OH', 'OW', 'OC', 'IC'],
)
@triton.jit
def conv3d_min_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr, D, H, W,
    OC: tl.constexpr, OD: tl.constexpr, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # grid: (N, ceil(OH*OW / BLOCK_HW))
    pid_n = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    s_total = OH * OW
    mask_s = offs < s_total
    oh = offs // OW
    ow = offs % OW

    oc_offs = tl.arange(0, BLOCK_OC)  # [BLOCK_OC]
    mask_oc = oc_offs < OC

    bias = tl.load(b_ptr + oc_offs, mask=mask_oc, other=0.0)  # [BLOCK_OC]

    INF = float('inf')
    # min_val [BLOCK_OC, BLOCK_HW]
    min_val = tl.full((BLOCK_OC, BLOCK_HW), INF, dtype=tl.float32)

    for od in range(0, OD):
        acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)
        for ic in range(0, IC):
            for kd in range(0, KD):
                id_ = od + kd
                for kh in range(0, KH):
                    ih = oh + kh  # [BLOCK_HW]
                    for kw in range(0, KW):
                        iw = ow + kw  # [BLOCK_HW]
                        x_off = (((pid_n * IC + ic) * D + id_) * H + ih) * W + iw
                        # weight: shape (OC, IC, KD, KH, KW)
                        w_off = ((ic * KD + kd) * KH + kh) * KW + kw + oc_offs * (IC * KD * KH * KW)
                        xv = tl.load(x_ptr + x_off, mask=mask_s, other=0.0)  # [BLOCK_HW]
                        wv = tl.load(w_ptr + w_off, mask=mask_oc, other=0.0)  # [BLOCK_OC]
                        acc += wv[:, None] * xv[None, :]
        acc = acc + bias[:, None]
        min_val = tl.minimum(min_val, acc)

    # softmax over OC dimension (axis=0)
    # mask invalid OC lanes to -inf
    min_val = tl.where(mask_oc[:, None], min_val, -INF)
    m = tl.max(min_val, axis=0)  # [BLOCK_HW]
    e = tl.exp(min_val - m[None, :])
    e = tl.where(mask_oc[:, None], e, 0.0)
    s = tl.sum(e, axis=0)  # [BLOCK_HW]
    out_v = e / s[None, :]

    # store: out shape (N, OC, OH*OW)
    out_off = (pid_n * OC + oc_offs[:, None]) * s_total + offs[None, :]
    store_mask = mask_oc[:, None] & mask_s[None, :]
    tl.store(out_ptr + out_off, out_v, mask=store_mask)


def _next_pow2(x):
    p = 1
    while p < x:
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
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, D, H, W = x.shape
        OC, _, KD, KH, KW = w.shape

        if self.dim == 2:
            OD = D - KD + 1
            OH = H - KH + 1
            OW = W - KW + 1

            out = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)
            BLOCK_OC = _next_pow2(OC)

            grid = lambda meta: (N, (OH * OW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'])
            conv3d_min_softmax_kernel[grid](
                x, w, b, out,
                N, IC, D, H, W,
                OC, OD, OH, OW,
                KD, KH, KW,
                BLOCK_OC=BLOCK_OC,
            )
            return out
        else:
            # fallback
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            y = torch.softmax(y, dim=1)
            return y