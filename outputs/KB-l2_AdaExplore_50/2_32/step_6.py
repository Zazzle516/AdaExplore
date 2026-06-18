import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
    ],
    key=['OC', 'OUT_HW', 'IC'],
)
@triton.jit
def conv_scale_min_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, H, W,
    OC, KH, KW,
    OH, OW,
    SCALE,
    OUT_HW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # tile along output spatial (OH*OW)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = offs_m < OUT_HW

    oh = offs_m // OW
    ow = offs_m % OW

    offs_ic = tl.arange(0, BLOCK_IC)  # = IC if BLOCK_IC==IC

    INF = float('inf')
    min_acc = tl.full([BLOCK_M], INF, dtype=tl.float32)

    # x is NHWC layout: x[n, h, w, ic]
    x_base = pid_n * H * W * IC

    num_oc_tiles = (OC + BLOCK_N - 1) // BLOCK_N
    for oc_tile in range(0, num_oc_tiles):
        offs_n = oc_tile * BLOCK_N + tl.arange(0, BLOCK_N)
        n_mask = offs_n < OC

        acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        # Iterate kernel positions (KH*KW small, e.g. 9)
        for kh in range(0, KH):
            for kw in range(0, KW):
                ih = oh + kh
                iw = ow + kw

                # Load x[pid_n, ih, iw, :IC] -> [BLOCK_M, BLOCK_IC]
                x_off = x_base + ih[:, None] * (W * IC) + iw[:, None] * IC + offs_ic[None, :]
                x_mask = m_mask[:, None]
                x_vals = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # Load w[oc, ic, kh, kw] for oc in offs_n -> [BLOCK_IC, BLOCK_N]
                # weight layout: w[OC, IC, KH, KW] contiguous
                w_off = (
                    offs_n[None, :] * (IC * KH * KW)
                    + offs_ic[:, None] * (KH * KW)
                    + kh * KW
                    + kw
                )
                w_mask = n_mask[None, :]
                w_vals = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals)

        bias = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
        acc = acc + bias[None, :]
        acc = acc * SCALE
        acc = tl.where(n_mask[None, :], acc, INF)

        tile_min = tl.min(acc, axis=1)
        min_acc = tl.minimum(min_acc, tile_min)

    out_offset = pid_n * OUT_HW + offs_m
    tl.store(out_ptr + out_offset, min_acc, mask=m_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = float(scale_factor)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        N, IC, H, W = x.shape
        OC, _, KH, KW = w.shape
        OH = H - KH + 1
        OW = W - KW + 1
        OUT_HW = OH * OW

        # Convert x to NHWC for coalesced ic-innermost access
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, 1, OH, OW), device=x.device, dtype=x.dtype)

        # BLOCK_IC must be a power of 2 >= IC for tl.dot; pick smallest pow2 >= IC
        BLOCK_IC = 1
        while BLOCK_IC < IC:
            BLOCK_IC *= 2

        grid = lambda meta: (N, (OUT_HW + meta['BLOCK_M'] - 1) // meta['BLOCK_M'])

        conv_scale_min_kernel[grid](
            x_nhwc, w, b, out,
            N, IC, H, W,
            OC, KH, KW,
            OH, OW,
            self.scale_factor,
            OUT_HW,
            BLOCK_IC=BLOCK_IC,
        )
        return out