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
    BLOCK_W: tl.constexpr,
):
    # one program per (n, dp, hp, wp_tile)
    pid = tl.program_id(0)
    # decode
    n_tiles_w = (Wp + BLOCK_W - 1) // BLOCK_W
    wt = pid % n_tiles_w
    tmp = pid // n_tiles_w
    hp = tmp % Hp
    tmp = tmp // Hp
    dp = tmp % Dp
    n = tmp // Dp

    d0 = dp * 2
    h0 = hp * 2
    wp_base = wt * BLOCK_W

    offs_c = tl.arange(0, BLOCK_C)
    offs_w = tl.arange(0, BLOCK_W)
    mask_c = offs_c < C
    mask_w = (wp_base + offs_w) < Wp

    # 2D shape: [BLOCK_C, BLOCK_W]
    c_idx = offs_c[:, None]
    w_idx = offs_w[None, :]
    w_pix = (wp_base + w_idx) * 2  # actual w position in input

    base = n * stride_n + c_idx * stride_c
    d_off0 = d0 * stride_d
    d_off1 = (d0 + 1) * stride_d
    h_off0 = h0 * stride_h
    h_off1 = (h0 + 1) * stride_h
    w_off0 = w_pix * stride_w
    w_off1 = (w_pix + 1) * stride_w

    mask2 = mask_c[:, None] & mask_w[None, :]
    NEG_INF = -float('inf')

    p000 = tl.load(x_ptr + base + d_off0 + h_off0 + w_off0, mask=mask2, other=NEG_INF)
    p001 = tl.load(x_ptr + base + d_off0 + h_off0 + w_off1, mask=mask2, other=NEG_INF)
    p010 = tl.load(x_ptr + base + d_off0 + h_off1 + w_off0, mask=mask2, other=NEG_INF)
    p011 = tl.load(x_ptr + base + d_off0 + h_off1 + w_off1, mask=mask2, other=NEG_INF)
    p100 = tl.load(x_ptr + base + d_off1 + h_off0 + w_off0, mask=mask2, other=NEG_INF)
    p101 = tl.load(x_ptr + base + d_off1 + h_off0 + w_off1, mask=mask2, other=NEG_INF)
    p110 = tl.load(x_ptr + base + d_off1 + h_off1 + w_off0, mask=mask2, other=NEG_INF)
    p111 = tl.load(x_ptr + base + d_off1 + h_off1 + w_off1, mask=mask2, other=NEG_INF)

    m1 = tl.maximum(p000, p001)
    m2 = tl.maximum(p010, p011)
    m3 = tl.maximum(p100, p101)
    m4 = tl.maximum(p110, p111)
    m5 = tl.maximum(m1, m2)
    m6 = tl.maximum(m3, m4)
    pooled = tl.maximum(m5, m6)  # [BLOCK_C, BLOCK_W]

    pooled_masked = tl.where(mask2, pooled, NEG_INF)
    row_max = tl.max(pooled_masked, axis=0)  # [BLOCK_W]
    exps = tl.exp(pooled_masked - row_max[None, :])
    exps = tl.where(mask2, exps, 0.0)
    sum_exp = tl.sum(exps, axis=0)  # [BLOCK_W]
    lse = tl.log(sum_exp) + row_max

    out_val = tl.maximum(lse, 0.0)

    out_offset = n * (Dp * Hp * Wp) + dp * (Hp * Wp) + hp * Wp + wp_base + offs_w
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

    BLOCK_W = 64 if Wp >= 64 else (32 if Wp >= 32 else 16)
    n_tiles_w = (Wp + BLOCK_W - 1) // BLOCK_W
    total = N * Dp * Hp * n_tiles_w
    grid = (total,)
    fused_pool_lse_relu_kernel[grid](
        x, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
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