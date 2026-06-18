import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 256, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64, 'BLOCK_OC': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 128, 'BLOCK_OC': 128}, num_warps=8, num_stages=3),
    ],
    key=['N_OUT', 'OC', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv_relu_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    K_TOTAL: tl.constexpr,
    N_OUT,  # B*OH*OW
    stride_xb, stride_xc, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wh, stride_ww,
    stride_yb, stride_yc, stride_yh, stride_yw,
    BLOCK_N: tl.constexpr, BLOCK_OC: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_k = tl.arange(0, BLOCK_K)

    mask_n = offs_n < N_OUT
    mask_oc = offs_oc < OC
    mask_k = offs_k < K_TOTAL

    # decompose n -> (b, oh, ow)
    ow = offs_n % OW
    tmp = offs_n // OW
    oh = tmp % OH
    b = tmp // OH

    # decompose k -> (ic, kh, kw): k = ic*KH*KW + kh*KW + kw
    kw_idx = offs_k % KW
    tmp_k = offs_k // KW
    kh_idx = tmp_k % KH
    ic_idx = tmp_k // KH

    # preload weights [BLOCK_OC, BLOCK_K]
    w_offs = (offs_oc[:, None] * stride_wo +
              ic_idx[None, :] * stride_wi +
              kh_idx[None, :] * stride_wh +
              kw_idx[None, :] * stride_ww)
    w_mask = mask_oc[:, None] & mask_k[None, :]
    w_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)  # [BLOCK_OC, BLOCK_K]

    # load x [BLOCK_N, BLOCK_K]
    ih = oh[:, None] + kh_idx[None, :]
    iw = ow[:, None] + kw_idx[None, :]
    x_offs = (b[:, None] * stride_xb +
              ic_idx[None, :] * stride_xc +
              ih * stride_xh +
              iw * stride_xw)
    x_mask = mask_n[:, None] & mask_k[None, :]
    x_tile = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)  # [BLOCK_N, BLOCK_K]

    # GEMM: [BLOCK_N, BLOCK_K] x [BLOCK_K, BLOCK_OC]
    acc = tl.dot(x_tile, tl.trans(w_tile))

    bias = tl.load(b_ptr + offs_oc, mask=mask_oc, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)
    # HardSwish-like: x * clamp((x+3)/6, 0, 1)
    hs = (acc + 3.0) / 6.0
    hs = tl.minimum(tl.maximum(hs, 0.0), 1.0)
    out = acc * hs

    # store
    y_offs = (b[:, None] * stride_yb + offs_oc[None, :] * stride_yc +
              oh[:, None] * stride_yh + ow[:, None] * stride_yw)
    mask = mask_n[:, None] & mask_oc[None, :]
    tl.store(y_ptr + y_offs, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous().cuda()
        b = self.conv.bias.contiguous().cuda()

        B, IC, IH, IW = x.shape
        OC, _, KH, KW = w.shape
        OH = IH - KH + 1
        OW = IW - KW + 1

        y = torch.empty((B, OC, OH, OW), device=x.device, dtype=x.dtype)

        N_OUT = B * OH * OW
        K_TOTAL = IC * KH * KW
        # round up to power of two >= 16
        BLOCK_K = 16
        while BLOCK_K < K_TOTAL:
            BLOCK_K *= 2

        grid = lambda meta: (
            triton.cdiv(N_OUT, meta['BLOCK_N']),
            triton.cdiv(OC, meta['BLOCK_OC']),
        )

        conv_relu_hardswish_kernel[grid](
            x, w, b, y,
            B, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            K_TOTAL,
            N_OUT,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            BLOCK_K=BLOCK_K,
        )
        return y