import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_lse_relu_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    Do, Ho, Wo,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_on, stride_od, stride_oh, stride_ow,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)

    num_w_blocks = (Wo + BLOCK_S - 1) // BLOCK_S
    blocks_per_n = Do * Ho * num_w_blocks

    pid_n = pid // blocks_per_n
    pid_in = pid - pid_n * blocks_per_n

    od = pid_in // (Ho * num_w_blocks)
    rem = pid_in - od * (Ho * num_w_blocks)
    oh = rem // num_w_blocks
    ow_blk = rem - oh * num_w_blocks

    ow_start = ow_blk * BLOCK_S
    offs_w = ow_start + tl.arange(0, BLOCK_S)
    mask_w = offs_w < Wo

    id0 = od * 2
    ih0 = oh * 2
    iw0 = offs_w * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    neg_inf = float('-inf')

    base = pid_n * stride_xn + offs_c[:, None] * stride_xc + iw0[None, :] * stride_xw

    full_mask = mask_c[:, None] & mask_w[None, :]

    # Load all 8 positions and take maximum
    p000 = tl.load(x_ptr + base + (id0 + 0) * stride_xd + (ih0 + 0) * stride_xh + 0 * stride_xw, mask=full_mask, other=neg_inf)
    p001 = tl.load(x_ptr + base + (id0 + 0) * stride_xd + (ih0 + 0) * stride_xh + 1 * stride_xw, mask=full_mask, other=neg_inf)
    p010 = tl.load(x_ptr + base + (id0 + 0) * stride_xd + (ih0 + 1) * stride_xh + 0 * stride_xw, mask=full_mask, other=neg_inf)
    p011 = tl.load(x_ptr + base + (id0 + 0) * stride_xd + (ih0 + 1) * stride_xh + 1 * stride_xw, mask=full_mask, other=neg_inf)
    p100 = tl.load(x_ptr + base + (id0 + 1) * stride_xd + (ih0 + 0) * stride_xh + 0 * stride_xw, mask=full_mask, other=neg_inf)
    p101 = tl.load(x_ptr + base + (id0 + 1) * stride_xd + (ih0 + 0) * stride_xh + 1 * stride_xw, mask=full_mask, other=neg_inf)
    p110 = tl.load(x_ptr + base + (id0 + 1) * stride_xd + (ih0 + 1) * stride_xh + 0 * stride_xw, mask=full_mask, other=neg_inf)
    p111 = tl.load(x_ptr + base + (id0 + 1) * stride_xd + (ih0 + 1) * stride_xh + 1 * stride_xw, mask=full_mask, other=neg_inf)

    m1 = tl.maximum(p000, p001)
    m2 = tl.maximum(p010, p011)
    m3 = tl.maximum(p100, p101)
    m4 = tl.maximum(p110, p111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    max_val = tl.maximum(m5, m6)

    masked_for_max = tl.where(mask_c[:, None], max_val, neg_inf)
    m = tl.max(masked_for_max, axis=0)
    shifted = max_val - m[None, :]
    e = tl.exp(shifted)
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) + m
    res = tl.maximum(lse, 0.0)

    out_idx = pid_n * stride_on + od * stride_od + oh * stride_oh + offs_w * stride_ow
    tl.store(out_ptr + out_idx, res, mask=mask_w)


def fused_pool_lse_relu(x):
    N, C, D, H, W = x.shape
    Do = D // 2
    Ho = H // 2
    Wo = W // 2
    out = torch.empty((N, 1, Do, Ho, Wo), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_S = 32

    num_w_blocks = (Wo + BLOCK_S - 1) // BLOCK_S
    grid = (N * Do * Ho * num_w_blocks,)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Do, Ho, Wo,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=8,
        num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_pool_lse_relu(x)
        return x