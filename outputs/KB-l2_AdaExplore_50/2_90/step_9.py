import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv3d_fused_kernel(
    x_ptr,          # [N, IC, ID, IH, IW]
    w_ptr,          # [OC, IC, KD, KH, KW]
    b_ptr,          # [OC] (conv bias + sum_tensor)
    out_ptr,        # [N, OC, OD, OH, OW]
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # OC tile
    BLOCK_N: tl.constexpr,   # spatial tile
    BLOCK_K: tl.constexpr,   # IC tile
    NEG_SLOPE: tl.constexpr,
):
    pid_n = tl.program_id(0)        # batch
    pid_m = tl.program_id(1)        # OC tile
    pid_s = tl.program_id(2)        # spatial tile

    OHW = OH * OW
    ODHW = OD * OHW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)        # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)        # spatial indices in [0, ODHW)

    m_mask = offs_m < OC
    s_mask = offs_s < ODHW

    # decode spatial indices
    od = offs_s // OHW
    rem = offs_s - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    # input base for this batch
    x_batch = x_ptr + pid_n * IC * ID * IH * IW

    # output base
    out_base = out_ptr + pid_n * OC * ODHW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KVOL = KD * KH * KW

    # loop over kernel positions and IC tiles
    for kk in range(0, KVOL):
        kd = kk // (KH * KW)
        kr = kk - kd * (KH * KW)
        kh = kr // KW
        kw = kr - kh * KW

        id_ = od + kd  # input depth idx
        ih_ = oh + kh
        iw_ = ow + kw

        # all valid since no padding and out dims = in - k + 1, but mask anyway
        spatial_in_offset = id_ * (IH * IW) + ih_ * IW + iw_  # [BLOCK_N]

        for ic_start in range(0, IC, BLOCK_K):
            offs_k = ic_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < IC

            # x: [BLOCK_K, BLOCK_N]
            x_offsets = offs_k[:, None] * (ID * IH * IW) + spatial_in_offset[None, :]
            x_load_mask = k_mask[:, None] & s_mask[None, :]
            x_vals = tl.load(x_batch + x_offsets, mask=x_load_mask, other=0.0)

            # w: [BLOCK_M, BLOCK_K]
            # w[oc, ic, kd, kh, kw] -> offset oc*(IC*KVOL) + ic*KVOL + kk
            w_offsets = offs_m[:, None] * (IC * KVOL) + offs_k[None, :] * KVOL + kk
            w_load_mask = m_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptr + w_offsets, mask=w_load_mask, other=0.0)

            acc += tl.dot(w_vals, x_vals)

    # add bias (conv bias + sum_tensor, per-OC)
    b_vals = tl.load(b_ptr + offs_m, mask=m_mask, other=0.0)  # [BLOCK_M]
    # leaky relu on conv output (before bias add? no: order is conv -> leaky -> +sum -> clamp -> gelu)
    # conv output includes conv bias already. So we need conv_bias added BEFORE leaky.
    # Therefore split: load conv_bias and sum_tensor separately.
    # We'll do this differently below.
    # (handled outside loop via separate pointers — see kernel rewrite)

    # Placeholder — overwritten by alternate code path
    tl.store(out_base + offs_m[:, None] * ODHW + offs_s[None, :],
             acc, mask=m_mask[:, None] & s_mask[None, :])


@triton.jit
def conv3d_fused_kernel_v2(
    x_ptr,
    w_ptr,
    cb_ptr,      # conv bias [OC]
    st_ptr,      # sum_tensor [OC]
    out_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NEG_SLOPE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_s = tl.program_id(2)

    OHW = OH * OW
    ODHW = OD * OHW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m < OC
    s_mask = offs_s < ODHW

    od = offs_s // OHW
    rem = offs_s - od * OHW
    oh = rem // OW
    ow = rem - oh * OW

    x_batch = x_ptr + pid_n * (IC * ID * IH * IW)
    out_base = out_ptr + pid_n * (OC * ODHW)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    KVOL: tl.constexpr = KD * KH * KW
    IDHW = ID * IH * IW
    IHW = IH * IW

    for kk in tl.static_range(0, KVOL):
        kd = kk // (KH * KW)
        kr = kk - kd * (KH * KW)
        kh = kr // KW
        kw = kr - kh * KW

        id_ = od + kd
        ih_ = oh + kh
        iw_ = ow + kw

        spatial_in_offset = id_ * IHW + ih_ * IW + iw_

        for ic_start in range(0, IC, BLOCK_K):
            offs_k = ic_start + tl.arange(0, BLOCK_K)
            k_mask = offs_k < IC

            x_offsets = offs_k[:, None] * IDHW + spatial_in_offset[None, :]
            x_load_mask = k_mask[:, None] & s_mask[None, :]
            x_vals = tl.load(x_batch + x_offsets, mask=x_load_mask, other=0.0)

            w_offsets = offs_m[:, None] * (IC * KVOL) + offs_k[None, :] * KVOL + kk
            w_load_mask = m_mask[:, None] & k_mask[None, :]
            w_vals = tl.load(w_ptr + w_offsets, mask=w_load_mask, other=0.0)

            acc += tl.dot(w_vals, x_vals)

    # bias
    cb = tl.load(cb_ptr + offs_m, mask=m_mask, other=0.0)
    st = tl.load(st_ptr + offs_m, mask=m_mask, other=0.0)
    acc = acc + cb[:, None]
    # leaky relu
    acc = tl.where(acc >= 0.0, acc, acc * NEG_SLOPE)
    # add sum_tensor
    acc = acc + st[:, None]
    # clamp
    acc = tl.minimum(tl.maximum(acc, -1.0), 1.0)
    # gelu exact
    inv_sqrt2 = 0.7071067811865475
    acc = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    out_offs = offs_m[:, None] * ODHW + offs_s[None, :]
    tl.store(out_base + out_offs, acc,
             mask=m_mask[:, None] & s_mask[None, :])


def conv3d_fused(x, weight, conv_bias, sum_tensor):
    x = x.contiguous()
    weight = weight.contiguous()
    conv_bias = conv_bias.contiguous()
    sum_flat = sum_tensor.contiguous().view(-1)

    N, IC, ID, IH, IW = x.shape
    OC, _, KD, KH, KW = weight.shape
    OD = ID - KD + 1
    OH = IH - KH + 1
    OW = IW - KW + 1

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 8 if IC <= 8 else 16

    grid = (N, triton.cdiv(OC, BLOCK_M), triton.cdiv(OD * OH * OW, BLOCK_N))

    conv3d_fused_kernel_v2[grid](
        x, weight, conv_bias, sum_flat, out,
        N, IC, ID, IH, IW,
        OC, OD, OH, OW,
        KD=KD, KH=KH, KW=KW,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NEG_SLOPE=0.2,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))

    def forward(self, x):
        x = x.contiguous()
        return conv3d_fused(x, self.conv.weight, self.conv.bias, self.sum_tensor)