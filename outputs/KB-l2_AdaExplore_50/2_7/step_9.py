import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OUT_SP', 'K'],
)
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr, w_ptr, conv_bias_ptr, extra_bias_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD, KH, KW,
    OUT_SP,  # OD*OH*OW
    K,        # IC*KD*KH*KW
    # strides for x: contiguous NCDHW
    x_stride_n, x_stride_c, x_stride_d, x_stride_h, x_stride_w,
    # strides for w: OC, IC, KD, KH, KW
    w_stride_oc, w_stride_ic, w_stride_kd, w_stride_kh, w_stride_kw,
    # strides for out: N, OC, OD, OH, OW
    o_stride_n, o_stride_oc, o_stride_od, o_stride_oh, o_stride_ow,
    BLOCK_M: tl.constexpr,  # OC tile
    BLOCK_N: tl.constexpr,  # spatial tile
    BLOCK_K: tl.constexpr,  # K reduction tile
):
    pid_n_batch = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)        # OC tile
    pid_s = tl.program_id(2)        # spatial tile

    n = pid_n_batch

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)  # spatial indices (out)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < OC
    mask_s = offs_s < OUT_SP

    # Decode spatial indices to (od, oh, ow)
    od = offs_s // (OH * OW)
    rem = offs_s % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    accumulator = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K = IC * KD * KH * KW
    KHKW = KH * KW
    KDKHKW = KD * KHKW

    for k_start in range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        mask_k = k_idx < K

        # Decode k -> (ic, kd, kh, kw)
        ic = k_idx // KDKHKW
        krem = k_idx % KDKHKW
        kd = krem // KHKW
        krem2 = krem % KHKW
        kh = krem2 // KW
        kw = krem2 % KW

        # Load weights: shape [BLOCK_M, BLOCK_K]  (oc, k)
        w_ptrs = (w_ptr
                  + offs_m[:, None] * w_stride_oc
                  + ic[None, :] * w_stride_ic
                  + kd[None, :] * w_stride_kd
                  + kh[None, :] * w_stride_kh
                  + kw[None, :] * w_stride_kw)
        w_mask = mask_m[:, None] & mask_k[None, :]
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Compute input spatial indices: id = od + kd, ih = oh + kh, iw = ow + kw (no padding, stride=1)
        id_ = od[:, None] + kd[None, :]   # [BLOCK_N, BLOCK_K] -> we want [BLOCK_K, BLOCK_N]
        ih_ = oh[:, None] + kh[None, :]
        iw_ = ow[:, None] + kw[None, :]

        # Build x_ptrs with shape [BLOCK_K, BLOCK_N]
        # transpose: indices are [BLOCK_N, BLOCK_K] -> we need [BLOCK_K, BLOCK_N]
        x_ptrs = (x_ptr
                  + n * x_stride_n
                  + ic[None, :] * x_stride_c
                  + id_.T * x_stride_d
                  + ih_.T * x_stride_h
                  + iw_.T * x_stride_w)
        x_mask = mask_k[:, None] & mask_s[None, :]
        x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0)

        accumulator += tl.dot(w_vals, x_vals)

    # Add conv bias
    cb = tl.load(conv_bias_ptr + offs_m, mask=mask_m, other=0.0)
    accumulator += cb[:, None]

    # Epilogue: ReLU -> (LeakyReLU is identity since x>=0) -> GELU -> Sigmoid -> + extra_bias
    x = tl.maximum(accumulator, 0.0)
    inv_sqrt2 = 0.7071067811865475
    g = 0.5 * x * (1.0 + tl.erf(x * inv_sqrt2))
    s = 1.0 / (1.0 + tl.exp(-g))

    eb = tl.load(extra_bias_ptr + offs_m, mask=mask_m, other=0.0)
    out = s + eb[:, None]

    # Store: out[n, oc, od, oh, ow]
    out_ptrs = (out_ptr
                + n * o_stride_n
                + offs_m[:, None] * o_stride_oc
                + od[None, :] * o_stride_od
                + oh[None, :] * o_stride_oh
                + ow[None, :] * o_stride_ow)
    out_mask = mask_m[:, None] & mask_s[None, :]
    tl.store(out_ptrs, out, mask=out_mask)


def fused_conv3d(x: torch.Tensor, weight: torch.Tensor, conv_bias: torch.Tensor, extra_bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    extra_bias = extra_bias.contiguous()

    N, IC, ID, IH, IW = x.shape
    OC, IC_w, KD, KH, KW = weight.shape
    assert IC == IC_w

    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    OUT_SP = OD * OH * OW
    K = IC * KD * KH * KW

    grid = lambda meta: (
        N,
        triton.cdiv(OC, meta['BLOCK_M']),
        triton.cdiv(OUT_SP, meta['BLOCK_N']),
    )

    conv3d_implicit_gemm_kernel[grid](
        x, weight, conv_bias, extra_bias, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        OUT_SP, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        weight = self.conv.weight
        conv_bias = self.conv.bias
        extra_bias = self.bias.view(-1).contiguous()
        return fused_conv3d(x, weight, conv_bias, extra_bias)