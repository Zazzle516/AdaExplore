import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


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
):
    # one program per (n, c, dp, hp)
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

    # load LN affine params (over W)
    w_ln = tl.load(weight_ptr + w_offs, mask=w_mask, other=0.0)
    b_ln = tl.load(bias_ptr + w_offs, mask=w_mask, other=0.0)

    inv_W = 1.0 / W

    # accumulator over pool window (per w position, length W)
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            d = d0 + dd
            h = h0 + hh
            base = ((n * C + c) * D + d) * H * W + h * W
            ptrs = x_ptr + base + w_offs
            vals = tl.load(ptrs, mask=w_mask, other=0.0).to(tl.float32)
            vals = vals + sum_weight
            # LN over W
            vals_m = tl.where(w_mask, vals, 0.0)
            mean = tl.sum(vals_m, axis=0) * inv_W
            diff = tl.where(w_mask, vals - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) * inv_W
            inv = 1.0 / tl.sqrt(var + eps)
            normed = diff * inv * w_ln + b_ln
            acc = acc + normed

    # acc has shape [BLOCK_W] containing sum of 4 normed rows; now pool over w pairs
    # pair (2k, 2k+1) -> output index k
    # use even/odd lanes
    even_mask = (w_offs % 2 == 0) & w_mask
    # We need to gather pairs. Easier: load again as two halves.
    # Alternative: create pooled output by indexing.
    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp

    # Load even and odd via separate pointer expressions referencing acc — but acc is a register tensor.
    # We need to extract acc[2*k] and acc[2*k+1]. Use tl.reshape if possible.
    acc2 = tl.reshape(acc, (BLOCK_W // 2, 2))
    pooled = tl.sum(acc2, axis=1)  # [BLOCK_W//2]
    pooled = pooled / 8.0

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

    out_base = ((n * C + c) * Dp + dp) * Hp * Wp + hp * Wp
    out_ptrs = out_ptr + out_base + wp_offs
    tl.store(out_ptrs, gelu, mask=wp_mask)


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
        num_warps=4,
        num_stages=2,
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