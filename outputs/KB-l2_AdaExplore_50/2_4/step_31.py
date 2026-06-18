import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Implicit GEMM conv2d with fused bias + double-mish epilogue.
# Layout: NHWC for input, output, weight as (OC, KH, KW, IC).
# One program per (N*OH*OW tile in M, OC tile in N).

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K', 'KH', 'KW', 'IC'],
)
@triton.jit
def conv_mish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    M, K_total,
    # strides (in elements)
    x_stride_n, x_stride_h, x_stride_w, x_stride_c,
    o_stride_n, o_stride_h, o_stride_w, o_stride_c,
    w_stride_oc, w_stride_kh, w_stride_kw, w_stride_ic,
    # GEMM dims
    N_DIM,  # = OC
    K,      # = IC * KH * KW
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # row in flattened (N*OH*OW)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channel

    # Decompose row into (n, oh, ow)
    ow = offs_m % OW
    tmp = offs_m // OW
    oh = tmp % OH
    n_idx = tmp // OH

    m_mask = offs_m < M
    n_mask = offs_n < N_DIM

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = IC * KH * KW
    offs_k = tl.arange(0, BLOCK_K)
    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        k_mask = k_idx < K

        # decompose k = ic * (KH*KW) + kh * KW + kw
        kw_i = k_idx % KW
        tmp_k = k_idx // KW
        kh_i = tmp_k % KH
        ic_i = tmp_k // KH

        # input positions
        ih = oh[:, None] + kh_i[None, :]  # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw_i[None, :]

        x_offsets = (n_idx[:, None] * x_stride_n
                     + ih * x_stride_h
                     + iw * x_stride_w
                     + ic_i[None, :] * x_stride_c)

        x_load_mask = m_mask[:, None] & k_mask[None, :]
        x_vals = tl.load(x_ptr + x_offsets, mask=x_load_mask, other=0.0)

        # weight: (OC, KH, KW, IC) -> w[oc, kh, kw, ic]
        w_offsets = (offs_n[None, :] * w_stride_oc
                     + kh_i[:, None] * w_stride_kh
                     + kw_i[:, None] * w_stride_kw
                     + ic_i[:, None] * w_stride_ic)
        w_load_mask = k_mask[:, None] & n_mask[None, :]
        w_vals = tl.load(w_ptr + w_offsets, mask=w_load_mask, other=0.0)

        acc += tl.dot(x_vals, w_vals)

    # Bias
    b = tl.load(b_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + b[None, :]

    # First mish: y = x * tanh(softplus(x))
    sp1 = tl.log(1.0 + tl.exp(acc))
    t1 = 2.0 * tl.sigmoid(2.0 * sp1) - 1.0
    y = acc * t1
    # Second mish
    sp2 = tl.log(1.0 + tl.exp(y))
    t2 = 2.0 * tl.sigmoid(2.0 * sp2) - 1.0
    z = y * t2

    # Store: out shape NHWC -> [n, oh, ow, oc]
    o_offsets = (n_idx[:, None] * o_stride_n
                 + oh[:, None] * o_stride_h
                 + ow[:, None] * o_stride_w
                 + offs_n[None, :] * o_stride_c)
    o_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + o_offsets, z, mask=o_mask)


def conv_mish_forward(x, weight, bias):
    # x: NCHW, weight: (OC, IC, KH, KW)
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1

    # NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # weight as (OC, KH, KW, IC)
    w_perm = weight.permute(0, 2, 3, 1).contiguous()

    out_nhwc = torch.empty((N, OH, OW, OC), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    K = IC * KH * KW
    N_DIM = OC

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_DIM, meta['BLOCK_N']),
    )

    conv_mish_kernel[grid](
        x_nhwc, w_perm, bias, out_nhwc,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        M, K,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
        w_perm.stride(0), w_perm.stride(1), w_perm.stride(2), w_perm.stride(3),
        N_DIM, K,
    )

    return out_nhwc.permute(0, 3, 1, 2).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        return conv_mish_forward(x, self.conv.weight, self.conv.bias)