import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    sum_w,
    eps,
    N, C, D, H, W,
    Dp, Hp, Wp,
    BLOCK_W: tl.constexpr,
):
    # one program: (n, c, dp, hp). Process the 2 d × 2 h rows = 4 rows of length W.
    pid = tl.program_id(0)
    hp = pid % Hp
    tmp = pid // Hp
    dp = tmp % Dp
    tmp = tmp // Dp
    c = tmp % C
    n = tmp // C

    w_offs = tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    gamma = tl.load(gamma_ptr + w_offs, mask=w_mask, other=0.0)
    beta = tl.load(beta_ptr + w_offs, mask=w_mask, other=0.0)

    inv_W = 1.0 / W

    base = n * (C * D * H * W) + c * (D * H * W)
    d0 = dp * 2
    h0 = hp * 2

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Manually unroll 4 rows
    # Row 0: (d0, h0)
    off0 = base + d0 * (H * W) + h0 * W + w_offs
    x0 = tl.load(x_ptr + off0, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    m0 = tl.sum(tl.where(w_mask, x0, 0.0), axis=0) * inv_W
    d0_ = tl.where(w_mask, x0 - m0, 0.0)
    v0 = tl.sum(d0_ * d0_, axis=0) * inv_W
    r0 = 1.0 / tl.sqrt(v0 + eps)
    n0 = (x0 - m0) * r0 * gamma + beta
    acc += n0

    # Row 1: (d0, h0+1)
    off1 = base + d0 * (H * W) + (h0 + 1) * W + w_offs
    x1 = tl.load(x_ptr + off1, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    m1 = tl.sum(tl.where(w_mask, x1, 0.0), axis=0) * inv_W
    d1_ = tl.where(w_mask, x1 - m1, 0.0)
    v1 = tl.sum(d1_ * d1_, axis=0) * inv_W
    r1 = 1.0 / tl.sqrt(v1 + eps)
    n1 = (x1 - m1) * r1 * gamma + beta
    acc += n1

    # Row 2: (d0+1, h0)
    off2 = base + (d0 + 1) * (H * W) + h0 * W + w_offs
    x2 = tl.load(x_ptr + off2, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    m2 = tl.sum(tl.where(w_mask, x2, 0.0), axis=0) * inv_W
    d2_ = tl.where(w_mask, x2 - m2, 0.0)
    v2 = tl.sum(d2_ * d2_, axis=0) * inv_W
    r2 = 1.0 / tl.sqrt(v2 + eps)
    n2 = (x2 - m2) * r2 * gamma + beta
    acc += n2

    # Row 3: (d0+1, h0+1)
    off3 = base + (d0 + 1) * (H * W) + (h0 + 1) * W + w_offs
    x3 = tl.load(x_ptr + off3, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    m3 = tl.sum(tl.where(w_mask, x3, 0.0), axis=0) * inv_W
    d3_ = tl.where(w_mask, x3 - m3, 0.0)
    v3 = tl.sum(d3_ * d3_, axis=0) * inv_W
    r3 = 1.0 / tl.sqrt(v3 + eps)
    n3 = (x3 - m3) * r3 * gamma + beta
    acc += n3

    # Pair-sum along W
    acc2 = tl.reshape(acc, (BLOCK_W // 2, 2))
    pair_sum = tl.sum(acc2, axis=1)
    pooled = pair_sum * (1.0 / 8.0)

    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp
    out_base = (((n * C + c) * Dp + dp) * Hp + hp) * Wp
    tl.store(out_ptr + out_base + wp_offs, gelu, mask=wp_mask)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.pool_kernel_size = tuple(pool_kernel_size) if isinstance(pool_kernel_size, (list, tuple)) else (pool_kernel_size,) * 3
        self.norm_shape = tuple(norm_shape) if isinstance(norm_shape, (list, tuple)) else (norm_shape,)

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape

        can_fuse = (
            len(self.norm_shape) == 1
            and self.norm_shape[0] == W
            and self.pool_kernel_size == (2, 2, 2)
            and D % 2 == 0 and H % 2 == 0 and W % 2 == 0
            and x.is_cuda and x.dtype == torch.float32
        )

        if can_fuse:
            x = x.contiguous()
            Dp = D // 2
            Hp = H // 2
            Wp = W // 2
            out = torch.empty((N, C, Dp, Hp, Wp), device=x.device, dtype=x.dtype)
            BLOCK_W = _next_pow2(W)
            grid = (N * C * Dp * Hp,)
            fused_post_kernel[grid](
                x, out,
                self.norm.weight, self.norm.bias,
                float(self.sum_weight.item()),
                float(self.norm.eps),
                N, C, D, H, W,
                Dp, Hp, Wp,
                BLOCK_W=BLOCK_W,
                num_warps=4,
                num_stages=2,
            )
            return out

        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x