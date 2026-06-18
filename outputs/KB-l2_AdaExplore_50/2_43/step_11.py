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
    OD, OH, OW,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, od, oh, ow_block); processes BLOCK_W outputs along W
    pid = tl.program_id(0)
    OW_B = OW // BLOCK_W
    owb = pid % OW_B
    tmp = pid // OW_B
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    d0 = od * 2
    h0 = oh * 2
    w0_base = owb * BLOCK_W * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    offs_w = tl.arange(0, BLOCK_W)  # [BLOCK_W]
    # input w coords for the 2 corners along W: w0 and w0+1
    w_left = w0_base + offs_w * 2  # [BLOCK_W]

    # base ptr: [BLOCK_W, BLOCK_C]
    base = n * stride_n + offs_c[None, :] * stride_c  # [1, BLOCK_C]
    w_off_l = w_left[:, None] * stride_w  # [BLOCK_W, 1]
    w_off_r = (w_left[:, None] + 1) * stride_w

    mask_2d = mask_c[None, :]

    p000 = in_ptr + base + (d0 + 0) * stride_d + (h0 + 0) * stride_h + w_off_l
    p001 = in_ptr + base + (d0 + 0) * stride_d + (h0 + 0) * stride_h + w_off_r
    p010 = in_ptr + base + (d0 + 0) * stride_d + (h0 + 1) * stride_h + w_off_l
    p011 = in_ptr + base + (d0 + 0) * stride_d + (h0 + 1) * stride_h + w_off_r
    p100 = in_ptr + base + (d0 + 1) * stride_d + (h0 + 0) * stride_h + w_off_l
    p101 = in_ptr + base + (d0 + 1) * stride_d + (h0 + 0) * stride_h + w_off_r
    p110 = in_ptr + base + (d0 + 1) * stride_d + (h0 + 1) * stride_h + w_off_l
    p111 = in_ptr + base + (d0 + 1) * stride_d + (h0 + 1) * stride_h + w_off_r

    neg_inf = float('-inf')
    v000 = tl.load(p000, mask=mask_2d, other=neg_inf)
    v001 = tl.load(p001, mask=mask_2d, other=neg_inf)
    v010 = tl.load(p010, mask=mask_2d, other=neg_inf)
    v011 = tl.load(p011, mask=mask_2d, other=neg_inf)
    v100 = tl.load(p100, mask=mask_2d, other=neg_inf)
    v101 = tl.load(p101, mask=mask_2d, other=neg_inf)
    v110 = tl.load(p110, mask=mask_2d, other=neg_inf)
    v111 = tl.load(p111, mask=mask_2d, other=neg_inf)

    m1 = tl.maximum(v000, v001)
    m2 = tl.maximum(v010, v011)
    m3 = tl.maximum(v100, v101)
    m4 = tl.maximum(v110, v111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    pooled = tl.maximum(m5, m6)  # [BLOCK_W, BLOCK_C]

    pooled_masked = tl.where(mask_2d, pooled, neg_inf)
    max_val = tl.max(pooled_masked, axis=1)  # [BLOCK_W]
    shifted = tl.where(mask_2d, pooled - max_val[:, None], neg_inf)
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(mask_2d, exp_vals, 0.0)
    sum_exp = tl.sum(exp_vals, axis=1)  # [BLOCK_W]
    lse = max_val + tl.log(sum_exp)
    res = tl.maximum(lse, 0.0)  # [BLOCK_W]

    out_w = owb * BLOCK_W + offs_w
    out_offset = n * out_stride_n + od * out_stride_d + oh * out_stride_h + out_w * out_stride_w
    tl.store(out_ptr + out_offset, res)


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

    BLOCK_W = 4
    if OW % BLOCK_W != 0:
        BLOCK_W = 2
        if OW % BLOCK_W != 0:
            BLOCK_W = 1

    grid = (N * OD * OH * (OW // BLOCK_W),)

    sN, sC, sD, sH, sW = x.stride()
    # out is contiguous [N,1,OD,OH,OW]
    o_sN = OD * OH * OW
    o_sD = OH * OW
    o_sH = OW
    o_sW = 1

    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        sN, sC, sD, sH, sW,
        o_sN, o_sD, o_sH, o_sW,
        BLOCK_C=BLOCK_C,
        BLOCK_W=BLOCK_W,
        num_warps=4,
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