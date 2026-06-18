import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
    ],
    key=['OC', 'OUT_SP', 'KVOL'],
)
@triton.jit
def conv3d_implicit_gemm_kernel(
    x_ptr, w_ptr, b_conv_ptr, b_extra_ptr, out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OUT_SP, KVOL,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # output spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
    BLOCK_K: tl.constexpr,  # IC tile
):
    pid_n = tl.program_id(0)  # batch
    pid_m = tl.program_id(1)  # output spatial tile
    pid_oc = tl.program_id(2)  # OC tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output spatial offsets
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)  # OC offsets
    offs_k = tl.arange(0, BLOCK_K)  # IC offsets

    m_mask = offs_m < OUT_SP
    n_mask = offs_n < OC

    # decompose output spatial -> (od, oh, ow)
    od = offs_m // (OH * OW)
    rem = offs_m % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over kernel positions and IC tiles
    for kd in tl.static_range(0, KD):
        for kh in tl.static_range(0, KH):
            for kw in tl.static_range(0, KW):
                id_ = od + kd
                ih = oh + kh
                iw = ow + kw
                # base x index for this batch and (id, ih, iw)
                x_base = pid_n * stride_xn + id_ * stride_xd + ih * stride_xh + iw * stride_xw  # [BLOCK_M]
                w_base = kd * stride_wd + kh * stride_wh + kw * stride_ww  # scalar offset

                for ic_start in range(0, IC, BLOCK_K):
                    ic_idx = ic_start + offs_k
                    k_mask = ic_idx < IC

                    # x: [BLOCK_M, BLOCK_K]
                    x_offs = x_base[:, None] + ic_idx[None, :] * stride_xc
                    x_mask = m_mask[:, None] & k_mask[None, :]
                    x_vals = tl.load(x_ptr + x_offs, mask=x_mask, other=0.0)

                    # w: [BLOCK_K, BLOCK_N]
                    w_offs = w_base + ic_idx[:, None] * stride_wi + offs_n[None, :] * stride_wo
                    w_mask = k_mask[:, None] & n_mask[None, :]
                    w_vals = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

                    acc += tl.dot(x_vals, w_vals, allow_tf32=True)

    # add conv bias
    bconv = tl.load(b_conv_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bconv[None, :]

    # epilogue: ReLU, GELU, Sigmoid, +extra bias
    acc = tl.maximum(acc, 0.0)
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))
    acc = tl.sigmoid(acc)
    bextra = tl.load(b_extra_ptr + offs_n, mask=n_mask, other=0.0)
    acc = acc + bextra[None, :]

    # store
    out_offs = (pid_n * stride_on
                + offs_n[None, :] * stride_oc
                + od[:, None] * stride_od
                + oh[:, None] * stride_oh
                + ow[:, None] * stride_ow)
    out_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


def fused_conv3d_act_bias(x, weight, bias_conv, bias_extra):
    x = x.contiguous()
    weight = weight.contiguous()
    bias_conv = bias_conv.contiguous()
    bias_extra = bias_extra.contiguous().view(-1)

    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1
    OUT_SP = OD * OH * OW
    KVOL = IC * KD * KH * KW

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_K = 8 if IC <= 8 else 16

    grid = lambda META: (
        N,
        triton.cdiv(OUT_SP, META['BLOCK_M']),
        triton.cdiv(OC, META['BLOCK_N']),
    )

    conv3d_implicit_gemm_kernel[grid](
        x, weight, bias_conv, bias_extra, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD, KH, KW,
        OUT_SP, KVOL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_K=BLOCK_K,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous()
        return fused_conv3d_act_bias(x, self.conv.weight, self.conv.bias, self.bias)