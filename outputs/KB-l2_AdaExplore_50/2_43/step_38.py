import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


torch.backends.cudnn.benchmark = True


@triton.jit
def fused_pool_lse_relu_kernel_nchw(
    in_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    wo = pid % Wo
    tmp = pid // Wo
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    n = tmp // Do

    d0 = do * 2
    h0 = ho * 2
    w0 = wo * 2

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    neg_inf = float('-inf')

    base = n * C * D * H * W
    chan_stride = D * H * W
    max_vals = tl.full([BLOCK_C], neg_inf, dtype=tl.float32)

    for dd in tl.static_range(2):
        for hh in tl.static_range(2):
            for ww in tl.static_range(2):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                offs = base + c_offs * chan_stride + d_idx * (H * W) + h_idx * W + w_idx
                v = tl.load(in_ptr + offs, mask=c_mask, other=neg_inf)
                max_vals = tl.maximum(max_vals, v)

    m = tl.max(tl.where(c_mask, max_vals, neg_inf), axis=0)
    exps = tl.where(c_mask, tl.exp(max_vals - m), 0.0)
    s = tl.sum(exps, axis=0)
    lse = m + tl.log(s)
    out_val = tl.maximum(lse, 0.0)

    out_offset = n * (Do * Ho * Wo) + do * (Ho * Wo) + ho * Wo + wo
    tl.store(out_ptr + out_offset, out_val)


@triton.jit
def fused_pool_lse_relu_kernel_nhwc(
    in_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    wo = pid % Wo
    tmp = pid // Wo
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    n = tmp // Do

    d0 = do * 2
    h0 = ho * 2
    w0 = wo * 2

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    neg_inf = float('-inf')

    # NDHWC layout: offset = n*D*H*W*C + d*H*W*C + h*W*C + w*C + c
    DHWC = D * H * W * C
    HWC = H * W * C
    WC = W * C

    base = n * DHWC
    max_vals = tl.full([BLOCK_C], neg_inf, dtype=tl.float32)

    for dd in tl.static_range(2):
        for hh in tl.static_range(2):
            for ww in tl.static_range(2):
                d_idx = d0 + dd
                h_idx = h0 + hh
                w_idx = w0 + ww
                offs = base + d_idx * HWC + h_idx * WC + w_idx * C + c_offs
                v = tl.load(in_ptr + offs, mask=c_mask, other=neg_inf)
                max_vals = tl.maximum(max_vals, v)

    m = tl.max(tl.where(c_mask, max_vals, neg_inf), axis=0)
    exps = tl.where(c_mask, tl.exp(max_vals - m), 0.0)
    s = tl.sum(exps, axis=0)
    lse = m + tl.log(s)
    out_val = tl.maximum(lse, 0.0)

    out_offset = n * (Do * Ho * Wo) + do * (Ho * Wo) + ho * Wo + wo
    tl.store(out_ptr + out_offset, out_val)


def fused_pool_lse_relu(x, channels_last=False):
    # x: (N, C, D, H, W)
    N, C, D, H, W = x.shape
    Do = D // 2
    Ho = H // 2
    Wo = W // 2
    out = torch.empty((N, 1, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * Do * Ho * Wo,)
    if channels_last:
        fused_pool_lse_relu_kernel_nhwc[grid](
            x, out,
            N, C, D, H, W,
            Do, Ho, Wo,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
    else:
        fused_pool_lse_relu_kernel_nchw[grid](
            x, out,
            N, C, D, H, W,
            Do, Ho, Wo,
            BLOCK_C=BLOCK_C,
            num_warps=4,
            num_stages=2,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv = self.conv.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        y = self.conv(x)
        # y is in channels_last_3d layout: physical NDHWC, logical NCDHW
        if y.is_contiguous(memory_format=torch.channels_last_3d):
            # Pass directly; kernel reads NDHWC physical layout
            out = fused_pool_lse_relu(y, channels_last=True)
        else:
            y = y.contiguous()
            out = fused_pool_lse_relu(y, channels_last=False)
        return out