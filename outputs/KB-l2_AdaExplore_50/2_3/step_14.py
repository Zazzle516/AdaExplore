import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Note from problem: Model defines LayerNorm(norm_shape=(out_channels,)).
# In PyTorch, LayerNorm with normalized_shape=(C,) normalizes over the LAST dim.
# But x after conv has shape [N, C, D, H, W]. LayerNorm(C,) expects last dim==C.
# Looking at the model: norm_shape = (out_channels,) = (64,) and x has last dim W=64
# Wait: W=32 here. But out_channels=64. The reference would fail at runtime then.
# Actually checking: stride=2 on input 32x32, with pad=1, output_padding=1, kernel=3:
#   out = (32-1)*2 - 2 + 3 + 1 = 62 - 2 + 3 + 1 = 64. So W=64, H=64, D=32.
# So last dim is 64 = out_channels. LayerNorm normalizes over W axis (last dim).
# The kernel in pool 0 already uses this assumption.


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
    BLOCK_W: tl.constexpr,
):
    # one program handles 2 hp positions for one (n, c, dp) => 4 LN rows reused
    pid = tl.program_id(0)
    # pid layout: (n, c, dp, hp_pair) where hp_pair in [0, Hp/?)
    # Simpler: keep one program per (n, c, dp, hp) — same as kernel 0
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

    w_ln = tl.load(weight_ptr + w_offs, mask=w_mask, other=0.0)
    b_ln = tl.load(bias_ptr + w_offs, mask=w_mask, other=0.0)

    inv_W = 1.0 / W

    # Load all 4 rows
    base0 = ((n * C + c) * D + d0) * H * W + h0 * W
    base1 = ((n * C + c) * D + d0) * H * W + (h0 + 1) * W
    base2 = ((n * C + c) * D + (d0 + 1)) * H * W + h0 * W
    base3 = ((n * C + c) * D + (d0 + 1)) * H * W + (h0 + 1) * W

    v0 = tl.load(x_ptr + base0 + w_offs, mask=w_mask, other=0.0).to(tl.float32) + sum_weight
    v1 = tl.load(x_ptr + base1 + w_offs, mask=w_mask, other=0.0).to(tl.float32) + sum_weight
    v2 = tl.load(x_ptr + base2 + w_offs, mask=w_mask, other=0.0).to(tl.float32) + sum_weight
    v3 = tl.load(x_ptr + base3 + w_offs, mask=w_mask, other=0.0).to(tl.float32) + sum_weight

    # LN each row over W
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
    pooled = tl.sum(acc2, axis=1) / 8.0

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp

    out_base = ((n * C + c) * Dp + dp) * Hp * Wp + hp * Wp
    tl.store(out_ptr + out_base + wp_offs, gelu, mask=wp_mask)


def fused_post(x, weight, bias, sum_weight, eps):
    N, C, D, H, W = x.shape
    Dp, Hp, Wp = D // 2, H // 2, W // 2
    out = torch.empty((N, C, Dp, Hp, Wp), dtype=x.dtype, device=x.device)
    BLOCK_W = triton.next_power_of_2(W)
    grid = (N * C * Dp * Hp,)
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