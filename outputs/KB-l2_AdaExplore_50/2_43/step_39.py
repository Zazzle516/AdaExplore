import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_pool_lse_relu_kernel(
    in_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW, OW_BLOCKS,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh, ow_block) where ow_block covers BLOCK_W output ws
    pid = tl.program_id(0)
    owb = pid % OW_BLOCKS
    tmp = pid // OW_BLOCKS
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    d0 = od * 2
    h0 = oh * 2
    ow_start = owb * BLOCK_W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_w = tl.arange(0, BLOCK_W)
    ow = ow_start + offs_w  # [BLOCK_W]
    mask_w = ow < OW
    w0 = ow * 2  # [BLOCK_W]

    # 2D shape: [BLOCK_C, BLOCK_W]
    base = n * stride_n + offs_c[:, None] * stride_c
    w_off0 = w0[None, :] * stride_w
    w_off1 = (w0[None, :] + 1) * stride_w

    d0s = d0 * stride_d
    d1s = (d0 + 1) * stride_d
    h0s = h0 * stride_h
    h1s = (h0 + 1) * stride_h

    mask = mask_c[:, None] & mask_w[None, :]
    neg_inf = float('-inf')

    v000 = tl.load(in_ptr + base + d0s + h0s + w_off0, mask=mask, other=neg_inf)
    v001 = tl.load(in_ptr + base + d0s + h0s + w_off1, mask=mask, other=neg_inf)
    v010 = tl.load(in_ptr + base + d0s + h1s + w_off0, mask=mask, other=neg_inf)
    v011 = tl.load(in_ptr + base + d0s + h1s + w_off1, mask=mask, other=neg_inf)
    v100 = tl.load(in_ptr + base + d1s + h0s + w_off0, mask=mask, other=neg_inf)
    v101 = tl.load(in_ptr + base + d1s + h0s + w_off1, mask=mask, other=neg_inf)
    v110 = tl.load(in_ptr + base + d1s + h1s + w_off0, mask=mask, other=neg_inf)
    v111 = tl.load(in_ptr + base + d1s + h1s + w_off1, mask=mask, other=neg_inf)

    m1 = tl.maximum(v000, v001)
    m2 = tl.maximum(v010, v011)
    m3 = tl.maximum(v100, v101)
    m4 = tl.maximum(v110, v111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    pooled = tl.maximum(m5, m6)  # [BLOCK_C, BLOCK_W]

    pooled_masked = tl.where(mask, pooled, neg_inf)
    max_val = tl.max(pooled_masked, axis=0)  # [BLOCK_W]
    shifted = tl.where(mask, pooled - max_val[None, :], neg_inf)
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(mask, exp_vals, 0.0)
    sum_exp = tl.sum(exp_vals, axis=0)  # [BLOCK_W]
    lse = max_val + tl.log(sum_exp)
    res = tl.maximum(lse, 0.0)

    out_offset = n * out_stride_n + od * out_stride_d + oh * out_stride_h + ow * out_stride_w
    tl.store(out_ptr + out_offset, res, mask=mask_w)


def fused_pool_lse_relu(x):
    # x: [N, C, D, H, W] -> output [N, 1, D//2, H//2, W//2]
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, D, H, W = x.shape
    OD, OH, OW = D // 2, H // 2, W // 2

    out = torch.empty((N, 1, OD, OH, OW), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C < 16:
        BLOCK_C = 16

    BLOCK_W = 16
    OW_BLOCKS = (OW + BLOCK_W - 1) // BLOCK_W

    grid = (N * OD * OH * OW_BLOCKS,)

    sN, sC, sD, sH, sW = x.stride()
    # out is contiguous [N,1,OD,OH,OW]
    o_sN = OD * OH * OW
    o_sD = OH * OW
    o_sH = OW
    o_sW = 1

    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW, OW_BLOCKS,
        sN, sC, sD, sH, sW,
        o_sN, o_sD, o_sH, o_sW,
        BLOCK_C=BLOCK_C,
        BLOCK_W=BLOCK_W,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_pool_lse_relu(x)
        return x