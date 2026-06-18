import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['N', 'C', 'D', 'H', 'W'],
)
@triton.jit
def fused_post_kernel(
    x_ptr,           # input after conv: [N, C, D, H, W]
    out_ptr,         # output: [N, C, D//2, H//2, W//2]
    weight_ptr,      # LayerNorm weight [W]
    bias_ptr,        # LayerNorm bias [W]
    sum_weight,      # scalar
    eps,
    N, C, D, H, W,
    Dp, Hp, Wp,
    HP_PER_PROG: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid layout: (n, c, dp, hp_block) where hp_block has HP_PER_PROG hp's
    hp_blocks = (Hp + HP_PER_PROG - 1) // HP_PER_PROG
    hpb = pid % hp_blocks
    tmp = pid // hp_blocks
    dp = tmp % Dp
    tmp = tmp // Dp
    c = tmp % C
    n = tmp // C

    d0 = dp * 2

    w_offs = tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    w_ln = tl.load(weight_ptr + w_offs, mask=w_mask, other=0.0)
    b_ln = tl.load(bias_ptr + w_offs, mask=w_mask, other=0.0)

    inv_W = 1.0 / W
    inv_sqrt2 = 0.7071067811865475

    nc_d0 = ((n * C + c) * D + d0) * H * W
    nc_d1 = ((n * C + c) * D + (d0 + 1)) * H * W

    out_nc_dp = ((n * C + c) * Dp + dp) * Hp * Wp
    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp

    for hh in tl.static_range(0, HP_PER_PROG):
        hp = hpb * HP_PER_PROG + hh
        do_work = hp < Hp
        h0 = hp * 2

        base0 = nc_d0 + h0 * W
        base1 = nc_d0 + (h0 + 1) * W
        base2 = nc_d1 + h0 * W
        base3 = nc_d1 + (h0 + 1) * W

        m_load = w_mask & do_work

        v0 = tl.load(x_ptr + base0 + w_offs, mask=m_load, other=0.0).to(tl.float32) + sum_weight
        v1 = tl.load(x_ptr + base1 + w_offs, mask=m_load, other=0.0).to(tl.float32) + sum_weight
        v2 = tl.load(x_ptr + base2 + w_offs, mask=m_load, other=0.0).to(tl.float32) + sum_weight
        v3 = tl.load(x_ptr + base3 + w_offs, mask=m_load, other=0.0).to(tl.float32) + sum_weight

        v0m = tl.where(w_mask, v0, 0.0)
        v1m = tl.where(w_mask, v1, 0.0)
        v2m = tl.where(w_mask, v2, 0.0)
        v3m = tl.where(w_mask, v3, 0.0)

        m0 = tl.sum(v0m, axis=0) * inv_W
        m1 = tl.sum(v1m, axis=0) * inv_W
        m2 = tl.sum(v2m, axis=0) * inv_W
        m3 = tl.sum(v3m, axis=0) * inv_W

        d0v = tl.where(w_mask, v0 - m0, 0.0)
        d1v = tl.where(w_mask, v1 - m1, 0.0)
        d2v = tl.where(w_mask, v2 - m2, 0.0)
        d3v = tl.where(w_mask, v3 - m3, 0.0)

        var0 = tl.sum(d0v * d0v, axis=0) * inv_W
        var1 = tl.sum(d1v * d1v, axis=0) * inv_W
        var2 = tl.sum(d2v * d2v, axis=0) * inv_W
        var3 = tl.sum(d3v * d3v, axis=0) * inv_W

        i0 = 1.0 / tl.sqrt(var0 + eps)
        i1 = 1.0 / tl.sqrt(var1 + eps)
        i2 = 1.0 / tl.sqrt(var2 + eps)
        i3 = 1.0 / tl.sqrt(var3 + eps)

        n0 = d0v * i0 * w_ln + b_ln
        n1 = d1v * i1 * w_ln + b_ln
        n2 = d2v * i2 * w_ln + b_ln
        n3 = d3v * i3 * w_ln + b_ln

        acc = n0 + n1 + n2 + n3

        acc2 = tl.reshape(acc, (BLOCK_W // 2, 2))
        pooled = tl.sum(acc2, axis=1) * 0.125

        gelu = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

        out_base = out_nc_dp + hp * Wp
        tl.store(out_ptr + out_base + wp_offs, gelu, mask=wp_mask & do_work)


def fused_post(x, weight, bias, sum_weight, eps):
    N, C, D, H, W = x.shape
    Dp, Hp, Wp = D // 2, H // 2, W // 2
    out = torch.empty((N, C, Dp, Hp, Wp), dtype=x.dtype, device=x.device)
    BLOCK_W = triton.next_power_of_2(W)
    HP_PER_PROG = 4
    hp_blocks = (Hp + HP_PER_PROG - 1) // HP_PER_PROG
    grid = (N * C * Dp * hp_blocks,)
    fused_post_kernel[grid](
        x, out, weight, bias,
        float(sum_weight), float(eps),
        N, C, D, H, W, Dp, Hp, Wp,
        HP_PER_PROG=HP_PER_PROG,
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
        if (isinstance(self.pool_kernel_size, tuple) and self.pool_kernel_size == (2, 2, 2)
                and self.norm.normalized_shape == (W,)
                and D % 2 == 0 and H % 2 == 0 and W % 2 == 0):
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