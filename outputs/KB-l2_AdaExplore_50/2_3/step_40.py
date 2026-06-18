import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


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
    pid = tl.program_id(0)
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


@triton.jit
def fused_post_kernel_batched(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    sum_w,
    eps,
    N, C, D, H, W,
    Dp, Hp, Wp,
    BLOCK_W: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,  # number of hp output rows per program
):
    # Each program handles ROWS_PER_PROG hp rows × all Wp simultaneously, processing
    # 4 input rows (d0,d1)x(h0,h1) per hp by batching them into a 2D tile.
    pid = tl.program_id(0)
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

    # Build a 2D tile of shape [4*ROWS_PER_PROG, BLOCK_W] for all rows.
    # Row order: for r in ROWS_PER_PROG: for dd in 0,1: for hh in 0,1:
    NUM_ROWS: tl.constexpr = 4 * ROWS_PER_PROG
    row_idx = tl.arange(0, NUM_ROWS)  # [NUM_ROWS]
    # decode r, dd, hh
    r_idx = row_idx // 4
    sub = row_idx % 4
    dd_idx = sub // 2
    hh_idx = sub % 2

    hp_per_row = hp_g * ROWS_PER_PROG + r_idx  # [NUM_ROWS]
    in_range = hp_per_row < Hp
    h0_per_row = hp_per_row * 2
    d_per_row = d0 + dd_idx
    h_per_row = h0_per_row + hh_idx

    # row offsets: [NUM_ROWS]
    row_base = nc_base + d_per_row * stride_d + h_per_row * stride_h
    # full offsets [NUM_ROWS, BLOCK_W]
    offs = row_base[:, None] + w_offs[None, :]
    full_mask = (w_mask[None, :]) & (in_range[:, None])

    x = tl.load(x_ptr + offs, mask=full_mask, other=0.0).to(tl.float32)
    x = x + sum_w

    # mean along W per row
    x_for_sum = tl.where(full_mask, x, 0.0)
    mean = tl.sum(x_for_sum, axis=1) * inv_W  # [NUM_ROWS]
    diff = tl.where(full_mask, x - mean[:, None], 0.0)
    var = tl.sum(diff * diff, axis=1) * inv_W
    rstd = 1.0 / tl.sqrt(var + eps)
    normed = (x - mean[:, None]) * rstd[:, None] * gamma[None, :] + beta[None, :]
    # normed: [NUM_ROWS, BLOCK_W]

    # Sum the 4 (dd,hh) rows for each r: reshape to [ROWS_PER_PROG, 4, BLOCK_W]
    normed_3d = tl.reshape(normed, (ROWS_PER_PROG, 4, BLOCK_W))
    summed = tl.sum(normed_3d, axis=1)  # [ROWS_PER_PROG, BLOCK_W]

    # Pool along W: reshape last dim into [BLOCK_W//2, 2] and sum
    pooled_3d = tl.reshape(summed, (ROWS_PER_PROG, BLOCK_W // 2, 2))
    pair_sum = tl.sum(pooled_3d, axis=2)  # [ROWS_PER_PROG, BLOCK_W//2]
    avg = pair_sum * 0.125

    gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))

    # Store
    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp
    r_arange = tl.arange(0, ROWS_PER_PROG)
    hp_per_out = hp_g * ROWS_PER_PROG + r_arange
    hp_mask = hp_per_out < Hp
    out_base_2d = (((n * C + c) * Dp + dp) * Hp + hp_per_out) * Wp  # [ROWS_PER_PROG]
    out_offs = out_base_2d[:, None] + wp_offs[None, :]
    out_mask = hp_mask[:, None] & wp_mask[None, :]
    tl.store(out_ptr + out_offs, gelu, mask=out_mask)


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
                fused_post_kernel_batched[grid](
                    x, out,
                    self.norm.weight, self.norm.bias,
                    float(self.sum_weight.item()),
                    float(self.norm.eps),
                    N, C, D, H, W, Dp, Hp, Wp,
                    BLOCK_W=BLOCK_W,
                    ROWS_PER_PROG=ROWS_PER_PROG,
                    num_warps=8,
                    num_stages=2,
                )
            else:
                ROWS_PER_PROG2 = 2 if Hp % 2 == 0 else 1
                hp_groups = (Hp + ROWS_PER_PROG2 - 1) // ROWS_PER_PROG2
                grid = (N * C * Dp * hp_groups,)
                fused_post_kernel_multi[grid](
                    x, out,
                    self.norm.weight, self.norm.bias,
                    float(self.sum_weight.item()),
                    float(self.norm.eps),
                    N, C, D, H, W, Dp, Hp, Wp,
                    BLOCK_W=BLOCK_W,
                    ROWS_PER_PROG=ROWS_PER_PROG2,
                    num_warps=4,
                    num_stages=2,
                )
            return out

        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x