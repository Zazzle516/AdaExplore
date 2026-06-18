import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,           # input: (N, C, D, H, W) after conv_transpose + sum_weight is added inside
    out_ptr,         # output: (N, C, Dp, Hp, Wp)
    gamma_ptr,       # (W,)  -- LayerNorm over last dim
    beta_ptr,        # (W,)
    sum_w,           # scalar
    eps,             # scalar
    N, C, D, H, W,
    Dp, Hp, Wp,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, c, dp, hp); each program processes Wp output positions
    pid = tl.program_id(0)
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

    base = ((n * C + c) * D + 0) * H * W  # base for (n,c,0,0,0)
    stride_d = H * W
    stride_h = W

    inv_W = 1.0 / W

    # accumulator across the 4 (d,h) positions for both even/odd w positions in pool
    # We'll compute 8 normalized vectors and average them. Then pool over W (pairs).
    # Pool over W means: out[wp] = avg over w in {2*wp, 2*wp+1} of normed_at_w
    # Since avg is linear: sum of all 8 normed vectors / 8, then pair-pool over W.

    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            d = d0 + dd
            h = h0 + hh
            off = base + d * stride_d + h * stride_h + w_offs
            x = tl.load(x_ptr + off, mask=w_mask, other=0.0).to(tl.float32)
            x = x + sum_w
            mean = tl.sum(tl.where(w_mask, x, 0.0), axis=0) * inv_W
            diff = tl.where(w_mask, x - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) * inv_W
            rstd = 1.0 / tl.sqrt(var + eps)
            normed = (x - mean) * rstd * gamma + beta
            acc += normed

    # acc is sum of 4 normalized vectors over (d,h). Now we need to also include
    # the 2 d-positions and 2 h-positions => already done (4 positions). For 2x2x2
    # pool we need 8 spatial positions and divide by 8. We have 4; we still need
    # to pool over W: pair (2*wp, 2*wp+1) and divide by 2 -> total /8.

    # avg over 4 (d,h) positions: divide by 4 later combined with W pooling /2 => /8
    # Now perform pair-wise pooling over W axis: out[wp] = (acc[2*wp] + acc[2*wp+1]) / 8
    # Write Wp outputs.
    wp_offs = tl.arange(0, BLOCK_W // 2) if False else tl.arange(0, BLOCK_W)
    # Use a simpler approach: shift and add
    # Build even and odd indexed values.
    # We'll create masks for even positions only.
    half = BLOCK_W // 2

    # Load even and odd separately by recomputing? Easier: shuffle via arithmetic.
    # acc has shape [BLOCK_W]. We want pairs (acc[0]+acc[1], acc[2]+acc[3], ...).
    # Use tl.reshape if possible — Triton supports reshape on power-of-2 shapes.
    acc2 = tl.reshape(acc, (half, 2))
    pooled = tl.sum(acc2, axis=1) * (1.0 / 8.0)

    # GELU exact
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

    wp_idx = tl.arange(0, half)
    wp_mask = wp_idx < Wp

    out_off = (((n * C + c) * Dp + dp) * Hp + hp) * Wp + wp_idx
    tl.store(out_ptr + out_off, gelu, mask=wp_mask)


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
        self.norm_shape = tuple(norm_shape) if not isinstance(norm_shape, int) else (norm_shape,)

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape

        # Our fused kernel handles: norm_shape == (W,), pool == (2,2,2), all dims even.
        can_fuse = (
            len(self.norm_shape) == 1
            and self.norm_shape[0] == W
            and tuple(self.pool_kernel_size) == (2, 2, 2)
            and D % 2 == 0 and H % 2 == 0 and W % 2 == 0
        )

        if can_fuse:
            x = x.contiguous()
            Dp, Hp, Wp = D // 2, H // 2, W // 2
            out = torch.empty((N, C, Dp, Hp, Wp), device=x.device, dtype=x.dtype)
            BLOCK_W = _next_pow2(W)
            grid = (N * C * Dp * Hp,)
            gamma = self.norm.weight
            beta = self.norm.bias
            eps = self.norm.eps
            sum_w = float(self.sum_weight.detach().item())
            fused_post_kernel[grid](
                x, out, gamma, beta,
                sum_w, eps,
                N, C, D, H, W,
                Dp, Hp, Wp,
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