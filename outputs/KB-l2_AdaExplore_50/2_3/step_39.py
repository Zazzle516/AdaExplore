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
    pid = tl.program_id(0)
    hp = pid % Hp
    tmp = pid // Hp
    dp = tmp % Dp
    tmp = tmp // Dp
    c = tmp % C
    n = tmp // C

    d0 = dp * 2
    h0 = hp * 2

    w_offs = tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    gamma = tl.load(gamma_ptr + w_offs, mask=w_mask, other=0.0)
    beta = tl.load(beta_ptr + w_offs, mask=w_mask, other=0.0)

    stride_d = H * W
    stride_h = W
    nc_base = (n * C + c) * D * H * W

    inv_W = 1.0 / W

    acc_row = tl.zeros([BLOCK_W], dtype=tl.float32)

    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            d = d0 + dd
            h = h0 + hh
            row_off = nc_base + d * stride_d + h * stride_h + w_offs
            x = tl.load(x_ptr + row_off, mask=w_mask, other=0.0).to(tl.float32)
            x = x + sum_w
            mean = tl.sum(tl.where(w_mask, x, 0.0), axis=0) * inv_W
            diff = tl.where(w_mask, x - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) * inv_W
            rstd = 1.0 / tl.sqrt(var + eps)
            normed = (x - mean) * rstd * gamma + beta
            acc_row += normed

    acc_2d = tl.reshape(acc_row, (BLOCK_W // 2, 2))
    pair_sum = tl.sum(acc_2d, axis=1)
    avg = pair_sum * 0.125

    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))

    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp
    out_base = (((n * C + c) * Dp + dp) * Hp + hp) * Wp
    tl.store(out_ptr + out_base + wp_offs, gelu, mask=wp_mask)


@triton.jit
def fused_post_kernel_multi(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    sum_w,
    eps,
    N, C, D, H, W,
    Dp, Hp, Wp,
    BLOCK_W: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    # Each program handles ROWS_PER_PROG output (hp) rows for fixed (n, c, dp).
    pid = tl.program_id(0)
    # Number of hp groups per (n,c,dp)
    hp_groups = (Hp + ROWS_PER_PROG - 1) // ROWS_PER_PROG
    hp_g = pid % hp_groups
    tmp = pid // hp_groups
    dp = tmp % Dp
    tmp = tmp // Dp
    c = tmp % C
    n = tmp // C

    d0 = dp * 2

    w_offs = tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    gamma = tl.load(gamma_ptr + w_offs, mask=w_mask, other=0.0)
    beta = tl.load(beta_ptr + w_offs, mask=w_mask, other=0.0)

    stride_d = H * W
    stride_h = W
    nc_base = (n * C + c) * D * H * W

    inv_W = 1.0 / W
    inv_sqrt2 = 0.70710678118654752440

    for r in tl.static_range(0, ROWS_PER_PROG):
        hp = hp_g * ROWS_PER_PROG + r
        # guard with mask via predicated stores
        in_range = hp < Hp
        h0 = hp * 2

        acc_row = tl.zeros([BLOCK_W], dtype=tl.float32)

        for dd in tl.static_range(0, 2):
            for hh in tl.static_range(0, 2):
                d = d0 + dd
                h = h0 + hh
                row_off = nc_base + d * stride_d + h * stride_h + w_offs
                x = tl.load(x_ptr + row_off, mask=w_mask & in_range, other=0.0).to(tl.float32)
                x = x + sum_w
                mean = tl.sum(tl.where(w_mask, x, 0.0), axis=0) * inv_W
                diff = tl.where(w_mask, x - mean, 0.0)
                var = tl.sum(diff * diff, axis=0) * inv_W
                rstd = 1.0 / tl.sqrt(var + eps)
                normed = (x - mean) * rstd * gamma + beta
                acc_row += normed

        acc_2d = tl.reshape(acc_row, (BLOCK_W // 2, 2))
        pair_sum = tl.sum(acc_2d, axis=1)
        avg = pair_sum * 0.125
        gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))

        wp_offs = tl.arange(0, BLOCK_W // 2)
        wp_mask = (wp_offs < Wp) & in_range
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
        self.pool_kernel_size = pool_kernel_size
        self.norm_shape = norm_shape

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape

        norm_dim_ok = (len(self.norm_shape) == 1 and self.norm_shape[0] == W)
        pool_ok = (tuple(self.pool_kernel_size) == (2, 2, 2)
                   and D % 2 == 0 and H % 2 == 0 and W % 2 == 0)

        if norm_dim_ok and pool_ok and x.is_cuda and x.dtype == torch.float32:
            x = x.contiguous()
            Dp, Hp, Wp = D // 2, H // 2, W // 2
            out = torch.empty((N, C, Dp, Hp, Wp), device=x.device, dtype=x.dtype)
            BLOCK_W = _next_pow2(W)
            if BLOCK_W < 2:
                BLOCK_W = 2

            ROWS_PER_PROG = 4
            if Hp % ROWS_PER_PROG == 0:
                hp_groups = Hp // ROWS_PER_PROG
                grid = (N * C * Dp * hp_groups,)
                fused_post_kernel_multi[grid](
                    x, out,
                    self.norm.weight, self.norm.bias,
                    float(self.sum_weight.item()),
                    float(self.norm.eps),
                    N, C, D, H, W, Dp, Hp, Wp,
                    BLOCK_W=BLOCK_W,
                    ROWS_PER_PROG=ROWS_PER_PROG,
                    num_warps=4,
                    num_stages=2,
                )
            else:
                grid = (N * C * Dp * Hp,)
                fused_post_kernel[grid](
                    x, out,
                    self.norm.weight, self.norm.bias,
                    float(self.sum_weight.item()),
                    float(self.norm.eps),
                    N, C, D, H, W, Dp, Hp, Wp,
                    BLOCK_W=BLOCK_W,
                    num_warps=4,
                )
            return out

        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x