import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.jit
def conv_epilogue_kernel(
    x_ptr,        # input NHWC: (N, H, W, IC)
    w_ptr,        # weight (KH*KW*IC, OC) contiguous
    b_ptr,        # bias (OC,)
    mult_ptr,     # multiplier (OC,)
    out_ptr,      # output NHWC: (N, OH, OW, OC) contiguous
    N, H, W, IC,
    OH, OW, OC,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # output spatial tile (OH*OW)
    BLOCK_N: tl.constexpr,   # OC tile
    BLOCK_K: tl.constexpr,   # IC tile (along K = KH*KW*IC)
    IC_C: tl.constexpr,      # IC as constexpr (assumed multiple of BLOCK_K)
):
    pid_m = tl.program_id(0)   # spatial tile id within image
    pid_n = tl.program_id(1)   # OC tile
    pid_b = tl.program_id(2)   # batch index

    OHW = OH * OW
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial offsets [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # oc offsets [BLOCK_N]

    mask_m = offs_m < OHW
    mask_n = offs_n < OC

    oh = offs_m // OW
    ow = offs_m % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # input batch base
    x_batch_ptr = x_ptr + pid_b * H * W * IC

    # Loop over KH, KW, and IC tiles
    for kh in tl.static_range(0, KH):
        ih = oh + kh  # no padding
        for kw in tl.static_range(0, KW):
            iw = ow + kw
            # x_row pointer for each m: base + ih*W*IC + iw*IC
            x_row_off = ih * (W * IC) + iw * IC   # [BLOCK_M]
            # weight row base for this (kh, kw): (kh*KW + kw) * IC, then add ic in K loop
            w_kbase = (kh * KW + kw) * IC

            for ic_start in tl.static_range(0, IC_C, BLOCK_K):
                offs_k = ic_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                # x: load [BLOCK_M, BLOCK_K]
                x_ptrs = x_batch_ptr + x_row_off[:, None] + offs_k[None, :]
                x_mask = mask_m[:, None]
                x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

                # w: load [BLOCK_K, BLOCK_N]
                w_ptrs = w_ptr + (w_kbase + offs_k)[:, None] * OC + offs_n[None, :]
                w_mask = mask_n[None, :]
                w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

                acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # epilogue: bias, multiplier, leaky_relu, gelu
    bias = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    mult = tl.load(mult_ptr + offs_n, mask=mask_n, other=0.0)
    y = acc + bias[None, :]
    y = y * mult[None, :]
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.70710678118654752440
    y = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    # store NHWC: out[b, oh, ow, oc]
    out_batch_ptr = out_ptr + pid_b * OH * OW * OC
    out_ptrs = out_batch_ptr + offs_m[:, None] * OC + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, y, mask=out_mask)


def fused_conv(x, weight_kkic_oc, bias, multiplier, KH, KW):
    # x: NHWC contiguous (N, H, W, IC)
    N, H, W, IC = x.shape
    OC = weight_kkic_oc.shape[1]
    OH = H - KH + 1
    OW = W - KW + 1

    out = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32

    grid = (triton.cdiv(OH * OW, BLOCK_M), triton.cdiv(OC, BLOCK_N), N)
    conv_epilogue_kernel[grid](
        x, weight_kkic_oc, bias, multiplier, out,
        N, H, W, IC, OH, OW, OC,
        KH=KH, KW=KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        IC_C=IC,
        num_warps=4, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._cached_weight = None
        self._cached_bias = None
        self._cached_mult = None

    def _prep(self):
        # weight: (OC, IC, KH, KW) -> (KH, KW, IC, OC) -> (KH*KW*IC, OC)
        w = self.conv.weight.detach()
        OC, IC, KH, KW = w.shape
        w_perm = w.permute(2, 3, 1, 0).contiguous().view(KH * KW * IC, OC)
        b = self.conv.bias.detach().contiguous()
        m = self.multiplier.detach().contiguous().view(-1)
        return w_perm, b, m, KH, KW

    def forward(self, x):
        # x: (N, C, H, W) -> NHWC
        x = x.contiguous()
        N, C, H, W = x.shape
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        w_perm, b, m, KH, KW = self._prep()

        out_nhwc = fused_conv(x_nhwc, w_perm, b, m, KH, KW)
        # back to NCHW
        out = out_nhwc.permute(0, 3, 1, 2).contiguous()
        return out