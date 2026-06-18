import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, b_ptr, m_ptr, out_ptr,
    N, IC, H, W,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program ids: (n, oc_tile, hw_tile)
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_m = pid_oc * BLOCK_M + tl.arange(0, BLOCK_M)  # OC dim
    offs_n = pid_hw * BLOCK_N + tl.arange(0, BLOCK_N)  # output spatial dim (flattened oh*ow)

    mask_m = offs_m < OC
    mask_n = offs_n < (OH * OW)

    oh = offs_n // OW
    ow = offs_n % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # x: [N, IC, H, W]; w: [OC, IC, KH, KW]
    # K = IC*KH*KW; iterate over kh, kw, ic (ic contiguous along K is best for w)
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # since padding=0, stride=1
            iw = ow + kw
            # Load weight slice [BLOCK_M, IC]: w[offs_m, :, kh, kw]
            w_base = offs_m[:, None] * (IC * KH * KW) + tl.arange(0, IC_C)[None, :] * (KH * KW) + (kh * KW + kw)
            w_mask = mask_m[:, None] & (tl.arange(0, IC_C)[None, :] < IC)
            w_tile = tl.load(w_ptr + w_base, mask=w_mask, other=0.0)  # [BLOCK_M, IC]

            # Load input slice [IC, BLOCK_N]: x[n, :, ih, iw]
            x_base = pid_n * (IC * H * W) + tl.arange(0, IC_C)[:, None] * (H * W) + ih[None, :] * W + iw[None, :]
            x_mask = (tl.arange(0, IC_C)[:, None] < IC) & mask_n[None, :]
            x_tile = tl.load(x_ptr + x_base, mask=x_mask, other=0.0)  # [IC, BLOCK_N]

            acc += tl.dot(w_tile, x_tile)

    # Add bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)  # [BLOCK_M]
    acc += bias[:, None]

    # Multiply by per-channel multiplier
    mult = tl.load(m_ptr + offs_m, mask=mask_m, other=0.0)  # [BLOCK_M]
    acc = acc * mult[:, None]

    # LeakyReLU(0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # Exact GELU
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: out[n, oc, oh, ow], layout [N, OC, OH, OW]
    out_offs = pid_n * (OC * OH * OW) + offs_m[:, None] * (OH * OW) + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Next power of 2 for IC
        ic = in_channels
        ic_c = 1
        while ic_c < ic:
            ic_c *= 2
        self.IC_C = ic_c

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        m = self.multiplier.contiguous().view(-1)

        N, IC, H, W = x.shape
        OC = w.shape[0]
        KH = w.shape[2]
        KW = w.shape[3]
        OH = H - KH + 1
        OW = W - KW + 1

        out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 64
        BLOCK_N = 128

        grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OH * OW, BLOCK_N))

        conv_fused_kernel[grid](
            x, w, b, m, out,
            N, IC, H, W,
            OC, OH, OW,
            KH, KW,
            self.IC_C,
            BLOCK_M, BLOCK_N,
            num_warps=4, num_stages=3,
        )
        return out