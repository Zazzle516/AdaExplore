import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=8, num_stages=2),
    ],
    key=['C', 'Wp'],
)
@triton.jit
def fused_pool_lse_relu_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    Dp, Hp, Wp,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # grid: (ceil(Wp/BLOCK_W), Hp, N*Dp)
    pid_w = tl.program_id(0)
    hp = tl.program_id(1)
    pid_nd = tl.program_id(2)
    n = pid_nd // Dp
    dp = pid_nd % Dp

    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_w = offs_w < Wp

    d0 = dp * 2
    h0 = hp * 2
    w0 = offs_w * 2  # [BLOCK_W]

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # base: [BLOCK_C], add w-offset [BLOCK_W] later
    base_nc = n * stride_n + offs_c * stride_c  # [BLOCK_C]
    base_dh = d0 * stride_d + h0 * stride_h
    # ptr per [BLOCK_W, BLOCK_C] = base_nc[None,:] + base_dh + w0[:,None]*stride_w
    base = base_nc[None, :] + base_dh + w0[:, None] * stride_w  # [BLOCK_W, BLOCK_C]

    sd = stride_d
    sh = stride_h
    sw = stride_w

    mc = mask_c[None, :] & mask_w[:, None]

    p000 = tl.load(x_ptr + base, mask=mc, other=-float('inf'))
    p001 = tl.load(x_ptr + base + sw, mask=mc, other=-float('inf'))
    p010 = tl.load(x_ptr + base + sh, mask=mc, other=-float('inf'))
    p011 = tl.load(x_ptr + base + sh + sw, mask=mc, other=-float('inf'))
    p100 = tl.load(x_ptr + base + sd, mask=mc, other=-float('inf'))
    p101 = tl.load(x_ptr + base + sd + sw, mask=mc, other=-float('inf'))
    p110 = tl.load(x_ptr + base + sd + sh, mask=mc, other=-float('inf'))
    p111 = tl.load(x_ptr + base + sd + sh + sw, mask=mc, other=-float('inf'))

    m1 = tl.maximum(p000, p001)
    m2 = tl.maximum(p010, p011)
    m3 = tl.maximum(p100, p101)
    m4 = tl.maximum(p110, p111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    pooled = tl.maximum(m5, m6)  # [BLOCK_W, BLOCK_C]

    pooled_masked = tl.where(mc, pooled, -float('inf'))
    row_max = tl.max(pooled_masked, axis=1)  # [BLOCK_W]
    exps = tl.exp(pooled_masked - row_max[:, None])
    exps = tl.where(mc, exps, 0.0)
    sum_exp = tl.sum(exps, axis=1)
    lse = tl.log(sum_exp) + row_max
    out_val = tl.maximum(lse, 0.0)

    out_offset = n * (Dp * Hp * Wp) + dp * (Hp * Wp) + hp * Wp + offs_w
    tl.store(out_ptr + out_offset, out_val, mask=mask_w)


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

    grid = lambda meta: ((Wp + meta['BLOCK_W'] - 1) // meta['BLOCK_W'], Hp, N * Dp)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        BLOCK_C=BLOCK_C,
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