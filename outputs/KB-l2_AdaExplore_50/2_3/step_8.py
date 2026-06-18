import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HP': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HP': 2}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HP': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HP': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HP': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HP': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HP': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HP': 4}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_HP': 8}, num_warps=4, num_stages=3),
    ],
    key=['N', 'C', 'D', 'H', 'W'],
)
@triton.jit
def fused_post_kernel(
    x_ptr,           # input after conv: [N, C, D, H, W]
    out_ptr,         # output after pool+gelu: [N, C, D//2, H//2, W//2]
    weight_ptr,      # LayerNorm weight [W]
    bias_ptr,        # LayerNorm bias [W]
    sum_weight,      # scalar
    eps,
    N, C, D, H, W,
    Dp, Hp, Wp,
    BLOCK_W: tl.constexpr,
    BLOCK_HP: tl.constexpr,
):
    # one program per (n, c, dp, hp_tile) with hp_tile covering BLOCK_HP output rows
    pid = tl.program_id(0)
    Hp_tiles = Hp // BLOCK_HP
    hp_tile = pid % Hp_tiles
    tmp = pid // Hp_tiles
    dp = tmp % Dp
    tmp = tmp // Dp
    c = tmp % C
    n = tmp // C

    d0 = dp * 2
    hp_start = hp_tile * BLOCK_HP
    h0 = hp_start * 2  # start of input H

    w_offs = tl.arange(0, BLOCK_W)
    w_mask = w_offs < W
    h_offs = tl.arange(0, BLOCK_HP * 2)  # input row offsets in this tile

    # load LN affine params (over W) once, reuse across dd
    w_ln = tl.load(weight_ptr + w_offs, mask=w_mask, other=0.0)
    b_ln = tl.load(bias_ptr + w_offs, mask=w_mask, other=0.0)

    inv_W = 1.0 / W

    # accumulator shape [BLOCK_HP*2, BLOCK_W]: sum of LN'd rows across dd
    acc = tl.zeros([BLOCK_HP * 2, BLOCK_W], dtype=tl.float32)

    for dd in tl.static_range(0, 2):
        d = d0 + dd
        base = ((n * C + c) * D + d) * H * W
        ptrs = x_ptr + base + (h0 + h_offs)[:, None] * W + w_offs[None, :]
        mask2d = w_mask[None, :]
        vals = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
        vals = vals + sum_weight
        vals_m = tl.where(mask2d, vals, 0.0)
        mean = tl.sum(vals_m, axis=1) * inv_W  # [BLOCK_HP*2]
        diff = tl.where(mask2d, vals - mean[:, None], 0.0)
        var = tl.sum(diff * diff, axis=1) * inv_W
        inv = 1.0 / tl.sqrt(var + eps)
        normed = diff * inv[:, None] * w_ln[None, :] + b_ln[None, :]
        acc = acc + normed

    # Pool over h-pairs: reshape to [BLOCK_HP, 2, BLOCK_W] and sum axis=1
    acc_r = tl.reshape(acc, (BLOCK_HP, 2, BLOCK_W))
    acc_h = tl.sum(acc_r, axis=1)  # [BLOCK_HP, BLOCK_W]
    # Pool over w-pairs: reshape to [BLOCK_HP, BLOCK_W//2, 2] and sum axis=2
    acc_w = tl.reshape(acc_h, (BLOCK_HP, BLOCK_W // 2, 2))
    pooled = tl.sum(acc_w, axis=2) * (1.0 / 8.0)  # [BLOCK_HP, BLOCK_W//2]

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

    hp_offs = hp_start + tl.arange(0, BLOCK_HP)
    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp
    out_base = ((n * C + c) * Dp + dp) * Hp * Wp
    out_ptrs = out_ptr + out_base + hp_offs[:, None] * Wp + wp_offs[None, :]
    tl.store(out_ptrs, gelu, mask=wp_mask[None, :])


def fused_post(x, weight, bias, sum_weight, eps):
    N, C, D, H, W = x.shape
    Dp, Hp, Wp = D // 2, H // 2, W // 2
    out = torch.empty((N, C, Dp, Hp, Wp), dtype=x.dtype, device=x.device)
    BLOCK_W = triton.next_power_of_2(W)
    def grid(meta):
        return (N * C * Dp * (Hp // meta['BLOCK_HP']),)
    fused_post_kernel[grid](
        x, out, weight, bias,
        float(sum_weight), float(eps),
        N, C, D, H, W, Dp, Hp, Wp,
        BLOCK_W=BLOCK_W,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.pool_kernel_size = pool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        # Use fused kernel only when norm normalizes over W axis and pool is 2x2x2
        if (isinstance(self.pool_kernel_size, tuple) and self.pool_kernel_size == (2, 2, 2)
                and self.norm.normalized_shape == (W,)
                and D % 2 == 0 and H % 2 == 0 and W % 2 == 0
                and (H // 2) % 8 == 0):
            return fused_post(
                x.contiguous(),
                self.norm.weight,
                self.norm.bias,
                self.sum_weight.item(),
                self.norm.eps,
            )
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x