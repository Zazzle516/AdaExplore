import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_lse_relu_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    Dp, Hp, Wp,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program handles BLOCK_S spatial outputs (along the flattened spatial axis)
    pid = tl.program_id(0)
    total = N * Dp * Hp * Wp

    offs_s = pid * BLOCK_S + tl.arange(0, BLOCK_S)
    mask_s = offs_s < total

    wp = offs_s % Wp
    tmp = offs_s // Wp
    hp = tmp % Hp
    tmp = tmp // Hp
    dp = tmp % Dp
    n = tmp // Dp

    d0 = dp * 2
    h0 = hp * 2
    w0 = wp * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # broadcast: [BLOCK_S, BLOCK_C]
    base = n[:, None] * stride_n + offs_c[None, :] * stride_c
    d0b = d0[:, None] * stride_d
    h0b = h0[:, None] * stride_h
    w0b = w0[:, None] * stride_w
    mc = mask_c[None, :] & mask_s[:, None]

    sd = stride_d
    sh = stride_h
    sw = stride_w

    p000 = tl.load(x_ptr + base + d0b + h0b + w0b, mask=mc, other=-float('inf'))
    p001 = tl.load(x_ptr + base + d0b + h0b + (w0b + sw), mask=mc, other=-float('inf'))
    p010 = tl.load(x_ptr + base + d0b + (h0b + sh) + w0b, mask=mc, other=-float('inf'))
    p011 = tl.load(x_ptr + base + d0b + (h0b + sh) + (w0b + sw), mask=mc, other=-float('inf'))
    p100 = tl.load(x_ptr + base + (d0b + sd) + h0b + w0b, mask=mc, other=-float('inf'))
    p101 = tl.load(x_ptr + base + (d0b + sd) + h0b + (w0b + sw), mask=mc, other=-float('inf'))
    p110 = tl.load(x_ptr + base + (d0b + sd) + (h0b + sh) + w0b, mask=mc, other=-float('inf'))
    p111 = tl.load(x_ptr + base + (d0b + sd) + (h0b + sh) + (w0b + sw), mask=mc, other=-float('inf'))

    m1 = tl.maximum(p000, p001)
    m2 = tl.maximum(p010, p011)
    m3 = tl.maximum(p100, p101)
    m4 = tl.maximum(p110, p111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    pooled = tl.maximum(m5, m6)  # [BLOCK_S, BLOCK_C]

    pooled_masked = tl.where(mc, pooled, -float('inf'))
    row_max = tl.max(pooled_masked, axis=1)  # [BLOCK_S]
    exps = tl.exp(pooled_masked - row_max[:, None])
    exps = tl.where(mc, exps, 0.0)
    sum_exp = tl.sum(exps, axis=1)
    lse = tl.log(sum_exp) + row_max
    out_val = tl.maximum(lse, 0.0)

    tl.store(out_ptr + offs_s, out_val, mask=mask_s)


def fused_pool_lse_relu(x):
    N, C, D, H, W = x.shape
    Dp = D // 2
    Hp = H // 2
    Wp = W // 2
    x = x.contiguous()
    out = torch.empty((N, 1, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    # next power of 2 for C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    total = N * Dp * Hp * Wp
    BLOCK_S = 64
    grid = ((total + BLOCK_S - 1) // BLOCK_S,)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C,
        BLOCK_S=BLOCK_S,
        num_warps=4,
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