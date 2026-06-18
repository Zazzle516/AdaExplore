import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,           # input: (N, C, D, H, W) after conv_transpose + sum_weight
    out_ptr,         # output: (N, C, Dp, Hp, Wp)
    gamma_ptr,       # (W,) - LayerNorm weight (over last dim)
    beta_ptr,        # (W,) - LayerNorm bias
    sum_w,           # scalar
    eps,             # scalar
    N, C, D, H, W,
    Dp, Hp, Wp,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, c, dp, hp) -- iterates over Wp positions
    pid = tl.program_id(0)
    # decode pid -> (n, c, dp, hp)
    hp = pid % Hp
    tmp = pid // Hp
    dp = tmp % Dp
    tmp = tmp // Dp
    c = tmp % C
    n = tmp // C

    d0 = dp * 2
    d1 = d0 + 1
    h0 = hp * 2
    h1 = h0 + 1

    w_offs = tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    gamma = tl.load(gamma_ptr + w_offs, mask=w_mask, other=0.0)
    beta = tl.load(beta_ptr + w_offs, mask=w_mask, other=0.0)

    # base offset to (n, c, *, *, *) row of length W
    base = ((n * C + c) * D + 0) * H * W  # not used directly
    stride_d = H * W
    stride_h = W
    nc_base = (n * C + c) * D * H * W

    inv_W = 1.0 / W

    # We need to load 8 rows of length W (one per (dd,hh,ww)... wait, ww is along W),
    # actually pool is 2x2x2 so we average over (dd, hh, ww). Each row spans W,
    # but we only sum 2 consecutive w's per output. So 4 rows (d0/d1, h0/h1) and
    # for each we use w pairs.
    # Approach: for each of the 4 (dd,hh) rows, load row of length W, layernorm it,
    # then for each output Wp position average 2 consecutive normalized values.

    # We'll accumulate the per-output averaged result for all Wp outputs in a 2D tile
    # but BLOCK_W spans the input W. Output Wp = W//2.
    # Instead: for each (dd,hh), compute normalized row, then add to accumulator
    # rows arranged as pairs.

    # Accumulator for normalized rows summed across 4 (dd,hh) combos
    acc_row = tl.zeros([BLOCK_W], dtype=tl.float32)

    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            d = d0 + dd
            h = h0 + hh
            row_off = nc_base + d * stride_d + h * stride_h + w_offs
            x = tl.load(x_ptr + row_off, mask=w_mask, other=0.0).to(tl.float32)
            x = x + sum_w
            # layernorm over W
            mean = tl.sum(tl.where(w_mask, x, 0.0), axis=0) * inv_W
            diff = tl.where(w_mask, x - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) * inv_W
            rstd = 1.0 / tl.sqrt(var + eps)
            normed = (x - mean) * rstd * gamma + beta
            acc_row += normed

    # acc_row holds sum of 4 normalized rows, each of length W.
    # Now average pool along W (kernel 2): for each output position wp,
    # avg = (acc_row[2*wp] + acc_row[2*wp+1]) / 8
    # We'll iterate Wp outputs.
    # Load even and odd indexed values
    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp
    # gather even and odd
    even_idx = wp_offs * 2
    odd_idx = wp_offs * 2 + 1
    # Use tl.load via memory? We have acc_row in registers. Use where-based gather:
    # Easier: split via reshape - but Triton supports reshape.
    acc_2d = tl.reshape(acc_row, (BLOCK_W // 2, 2))
    pair_sum = tl.sum(acc_2d, axis=1)  # shape [BLOCK_W//2]
    avg = pair_sum * 0.125  # divide by 8

    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * avg * (1.0 + tl.erf(avg * inv_sqrt2))

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

        # Conditions for fast path:
        # - LayerNorm normalizes only the last dim and matches W
        # - pool kernel is (2,2,2) and dims divisible by 2
        norm_dim_ok = (len(self.norm_shape) == 1 and self.norm_shape[0] == W)
        pool_ok = (self.pool_kernel_size == (2, 2, 2) and D % 2 == 0
                   and H % 2 == 0 and W % 2 == 0)

        if norm_dim_ok and pool_ok and x.is_cuda and x.dtype == torch.float32:
            x = x.contiguous()
            Dp, Hp, Wp = D // 2, H // 2, W // 2
            out = torch.empty((N, C, Dp, Hp, Wp), device=x.device, dtype=x.dtype)
            BLOCK_W = _next_pow2(W)
            # ensure BLOCK_W >= 2 and a power of 2
            if BLOCK_W < 2:
                BLOCK_W = 2
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

        # Fallback
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x