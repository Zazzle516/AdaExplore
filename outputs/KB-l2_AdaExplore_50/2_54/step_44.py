import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_im2col_gemm_kernel(
    x_ptr,           # input NHWC: (N, H, W, IC)
    w_ptr,           # weight packed: (K = IC*KH*KW, OC)
    bias_ptr,        # (OC,)
    mult_ptr,        # (OC,)
    out_ptr,         # output NHWC: (N, OH, OW, OC)
    N, H, W, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial index
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel index

    M = N * OH * OW

    # decompose offs_m into (n, oh, ow)
    n_idx = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh_idx = rem // OW
    ow_idx = rem % OW

    mask_m = offs_m < M
    mask_n = offs_n < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = IC*KH*KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # decompose k into (ic, kh, kw)
        kw_i = offs_k % KW
        khc = offs_k // KW
        kh_i = khc % KH
        ic_i = khc // KH

        # input spatial coords for each (m, k)
        ih = oh_idx[:, None] + kh_i[None, :]   # (BLOCK_M, BLOCK_K)
        iw = ow_idx[:, None] + kw_i[None, :]
        nn_ = n_idx[:, None]                   # (BLOCK_M, 1)
        ic_ = ic_i[None, :]                    # (1, BLOCK_K)

        # input offset NHWC
        x_off = ((nn_ * H + ih) * W + iw) * IC + ic_
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

        # weight offset (K, OC)
        w_off = offs_k[:, None] * OC + offs_n[None, :]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    # bias
    b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + b[None, :]

    # multiplier (per output channel)
    m_ = tl.load(mult_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc * m_[None, :]

    # LeakyReLU(0.01)
    acc = tl.where(acc >= 0, acc, acc * 0.01)

    # GELU exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store NHWC
    out_off = offs_m[:, None] * OC + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)


def fused_conv(x, weight_packed, bias, multiplier, KH, KW, OC):
    # x: (N, H, W, IC) contiguous NHWC
    N, H, W, IC = x.shape
    OH = H - KH + 1
    OW = W - KW + 1
    K = IC * KH * KW

    out = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)
    M = N * OH * OW

    BLOCK_M = 128
    BLOCK_N = 64
    BLOCK_K = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(OC, BLOCK_N))

    conv_im2col_gemm_kernel[grid](
        x, weight_packed, bias, multiplier, out,
        N, H, W, IC,
        OH, OW, OC,
        KH=KH, KW=KW, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # x: (N, IC, H, W) -> NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        # weight: (OC, IC, KH, KW) -> packed (IC*KH*KW, OC)
        w = self.conv.weight  # (OC, IC, KH, KW)
        OC, IC, KH, KW = w.shape
        # Pack: for each (ic, kh, kw, oc) -> row-major (ic*KH*KW + kh*KW + kw, oc)
        w_packed = w.permute(1, 2, 3, 0).contiguous().view(IC * KH * KW, OC)

        bias = self.conv.bias
        mult = self.multiplier.contiguous().view(-1)  # (OC,)

        out_nhwc = fused_conv(x_nhwc, w_packed, bias, mult, KH, KW, OC)

        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out