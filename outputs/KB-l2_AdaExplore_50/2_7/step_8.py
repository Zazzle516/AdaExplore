import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_SPATIAL', 'K_TOTAL'],
)
@triton.jit
def conv3d_fused_kernel(
    x_ptr, w_ptr, conv_bias_ptr, post_bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    OUT_SPATIAL, K_TOTAL,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # reduction tile (over IC*KD*KH*KW)
):
    pid_n = tl.program_id(0)         # batch
    pid_m = tl.program_id(1)         # OC tile id
    pid_s = tl.program_id(2)         # spatial tile id

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices

    # Decompose spatial into (od, oh, ow)
    od = offs_s // (OH * OW)
    rem = offs_s - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    m_mask = offs_m < OC
    s_mask = offs_s < OUT_SPATIAL

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute base x pointer for the spatial tile (output positions)
    x_base = (x_ptr
              + pid_n * stride_xn
              + od * stride_xd
              + oh * stride_xh
              + ow * stride_xw)  # [BLOCK_N]

    # Loop over K = IC * KD * KH * KW
    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_TOTAL

        # Decompose k -> (ic, kd, kh, kw)
        kw_idx = offs_k % KW
        tmp1 = offs_k // KW
        kh_idx = tmp1 % KH
        tmp2 = tmp1 // KH
        kd_idx = tmp2 % KD
        ic_idx = tmp2 // KD

        # Weight: [OC, IC, KD, KH, KW]
        w_ptrs = (w_ptr
                  + offs_m[:, None] * stride_wo
                  + ic_idx[None, :] * stride_wi
                  + kd_idx[None, :] * stride_wd
                  + kh_idx[None, :] * stride_wh
                  + kw_idx[None, :] * stride_ww)
        w_load_mask = m_mask[:, None] & k_mask[None, :]
        w = tl.load(w_ptrs, mask=w_load_mask, other=0.0)

        # Per-k offset relative to output base
        k_off = (ic_idx * stride_xc
                 + kd_idx * stride_xd
                 + kh_idx * stride_xh
                 + kw_idx * stride_xw)  # [BLOCK_K]

        x_ptrs = x_base[None, :] + k_off[:, None]
        x_load_mask = k_mask[:, None] & s_mask[None, :]
        x = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

        acc += tl.dot(w, x)

    # Add conv bias
    cb = tl.load(conv_bias_ptr + offs_m, mask=m_mask, other=0.0)
    acc = acc + cb[:, None]

    # ReLU (also makes LeakyReLU a no-op)
    acc = tl.maximum(acc, 0.0)

    # GELU (exact)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # Sigmoid
    acc = 1.0 / (1.0 + tl.exp(-acc))

    # Post bias [OC]
    pb = tl.load(post_bias_ptr + offs_m, mask=m_mask, other=0.0)
    acc = acc + pb[:, None]

    # Store: out[n, oc, od, oh, ow]
    out_ptrs = (out_ptr
                + pid_n * stride_on
                + offs_m[:, None] * stride_oc
                + od[None, :] * stride_od
                + oh[None, :] * stride_oh
                + ow[None, :] * stride_ow)
    store_mask = m_mask[:, None] & s_mask[None, :]
    tl.store(out_ptrs, acc, mask=store_mask)


def conv3d_fused(x, w, conv_bias, post_bias):
    x = x.contiguous()
    w = w.contiguous()
    conv_bias = conv_bias.contiguous()
    post_bias = post_bias.contiguous()

    N, IC, ID, IH, IW = x.shape
    OC, ICw, KD, KH, KW = w.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    OUT_SPATIAL = OD * OH * OW
    K_TOTAL = IC * KD * KH * KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OUT_SPATIAL, meta['BLOCK_N']),
    )

    conv3d_fused_kernel[grid](
        x, w, conv_bias, post_bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        OUT_SPATIAL, K_TOTAL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3), w.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        w = self.conv.weight
        cb = self.conv.bias if self.conv.bias is not None else torch.zeros(
            w.shape[0], device=x.device, dtype=x.dtype)
        pb = self.bias.view(-1).contiguous()
        return conv3d_fused(x, w, cb, pb)