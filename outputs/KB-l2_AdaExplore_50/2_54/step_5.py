import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_fused_kernel(
    x_ptr, w_ptr, b_ptr, mult_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)  # N index
    pid_oc = tl.program_id(1)  # OC tile
    pid_m = tl.program_id(2)  # OH*OW tile

    # M dimension: OH*OW (spatial output positions for this N)
    M = OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]  -> output channel

    oh = offs_m // OW
    ow = offs_m % OW
    mask_m = offs_m < M
    mask_n = offs_n < OC

    # Weight layout: (OC, IC, KH, KW). We'll iterate over K = IC*KH*KW.
    # For better coalescing, we treat K as (kh, kw, ic) with ic innermost.
    # offsets:
    # x[n, ic, oh+kh, ow+kw]  stride: ic*IH*IW + (oh+kh)*IW + (ow+kw)
    # w[oc, ic, kh, kw]       stride: oc*IC*KH*KW + ic*KH*KW + kh*KW + kw

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Input base pointer for this N
    x_n_base = x_ptr + pid_n * IC * IH * IW

    # iterate over kh, kw, then chunks of IC
    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # [BLOCK_M]
            iw = ow + kw  # [BLOCK_M]
            # since padding=0, all in bounds (assuming valid output)
            for ic_start in range(0, IC, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                mask_k = offs_k < IC

                # Load x[n, ic, ih, iw] -> shape [BLOCK_M, BLOCK_K]
                x_offs = (offs_k[None, :] * IH * IW
                          + ih[:, None] * IW
                          + iw[:, None])
                x_vals = tl.load(x_n_base + x_offs,
                                 mask=mask_m[:, None] & mask_k[None, :],
                                 other=0.0)

                # Load w[oc, ic, kh, kw] -> shape [BLOCK_K, BLOCK_N]
                w_offs = (offs_n[None, :] * IC * KH * KW
                          + offs_k[:, None] * KH * KW
                          + kh * KW + kw)
                w_vals = tl.load(w_ptr + w_offs,
                                 mask=mask_k[:, None] & mask_n[None, :],
                                 other=0.0)

                acc += tl.dot(x_vals, w_vals)

    # Bias add
    b = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc = acc + b[None, :]

    # Multiplier per output channel
    m = tl.load(mult_ptr + offs_n, mask=mask_n, other=0.0)  # [BLOCK_N]
    acc = acc * m[None, :]

    # LeakyReLU (slope 0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Store: out[n, oc, oh, ow]
    out_n_base = out_ptr + pid_n * OC * OH * OW
    out_offs = offs_n[None, :] * OH * OW + offs_m[:, None]
    tl.store(out_n_base + out_offs, acc,
             mask=mask_m[:, None] & mask_n[None, :])


def fused_conv(x, weight, bias, multiplier):
    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    multiplier = multiplier.contiguous().view(-1)

    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (N, triton.cdiv(OC, BLOCK_N), triton.cdiv(OH * OW, BLOCK_M))

    conv_fused_kernel[grid](
        x, weight, bias, multiplier, out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW, IC,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        return fused_conv(x, self.conv.weight, self.conv.bias, self.multiplier)