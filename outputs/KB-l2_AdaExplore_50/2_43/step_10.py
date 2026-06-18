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
):
    # one program per (n, od, oh, ow)
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    d0 = od * 2
    h0 = oh * 2
    w0 = ow * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * stride_n + offs_c * stride_c

    # 8 corners of the 2x2x2 window
    v000 = tl.load(in_ptr + base + (d0 + 0) * stride_d + (h0 + 0) * stride_h + (w0 + 0) * stride_w, mask=mask_c, other=-float('inf'))
    v001 = tl.load(in_ptr + base + (d0 + 0) * stride_d + (h0 + 0) * stride_h + (w0 + 1) * stride_w, mask=mask_c, other=-float('inf'))
    v010 = tl.load(in_ptr + base + (d0 + 0) * stride_d + (h0 + 1) * stride_h + (w0 + 0) * stride_w, mask=mask_c, other=-float('inf'))
    v011 = tl.load(in_ptr + base + (d0 + 0) * stride_d + (h0 + 1) * stride_h + (w0 + 1) * stride_w, mask=mask_c, other=-float('inf'))
    v100 = tl.load(in_ptr + base + (d0 + 1) * stride_d + (h0 + 0) * stride_h + (w0 + 0) * stride_w, mask=mask_c, other=-float('inf'))
    v101 = tl.load(in_ptr + base + (d0 + 1) * stride_d + (h0 + 0) * stride_h + (w0 + 1) * stride_w, mask=mask_c, other=-float('inf'))
    v110 = tl.load(in_ptr + base + (d0 + 1) * stride_d + (h0 + 1) * stride_h + (w0 + 0) * stride_w, mask=mask_c, other=-float('inf'))
    v111 = tl.load(in_ptr + base + (d0 + 1) * stride_d + (h0 + 1) * stride_h + (w0 + 1) * stride_w, mask=mask_c, other=-float('inf'))

    # max over the 8 corners per channel
    m1 = tl.maximum(v000, v001)
    m2 = tl.maximum(v010, v011)
    m3 = tl.maximum(v100, v101)
    m4 = tl.maximum(v110, v111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    pooled = tl.maximum(m5, m6)  # [BLOCK_C], pooled per channel

    # logsumexp across channels
    neg_inf = float('-inf')
    pooled_masked = tl.where(mask_c, pooled, neg_inf)
    max_val = tl.max(pooled_masked, axis=0)
    shifted = tl.where(mask_c, pooled - max_val, neg_inf)
    exp_vals = tl.exp(shifted)
    exp_vals = tl.where(mask_c, exp_vals, 0.0)
    sum_exp = tl.sum(exp_vals, axis=0)
    lse = max_val + tl.log(sum_exp)

    # relu
    res = tl.maximum(lse, 0.0)

    out_offset = n * out_stride_n + od * out_stride_d + oh * out_stride_h + ow * out_stride_w
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

    grid = (N * OD * OH * OW,)

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
        num_warps=2,
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