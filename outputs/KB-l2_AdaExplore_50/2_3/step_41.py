import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,           # input: (N, C, D, H, W) after conv_transpose + sum_weight already added? No, we add inside.
    out_ptr,         # output: (N, C, Dp, Hp, Wp)
    gamma_ptr,       # (W_norm,) LayerNorm weight (last dim)
    beta_ptr,        # (W_norm,)
    sum_w,           # scalar
    eps,             # scalar
    N, C, D, H, W,
    Dp, Hp, Wp,
    BLOCK_W: tl.constexpr,
):
    # One program handles one (n, c, dp, hp) and the full W row (for 2 H rows, 2 D rows).
    # LayerNorm is over the last dim (W,). Each (n, c, d, h) row is normalized independently.
    # Then average pool 2x2x2 -> one output row of size Wp at (n, c, dp, hp).
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

    # Compute base offset for (n, c, *, *, *)
    base = n * (C * D * H * W) + c * (D * H * W)

    # Accumulator: average over 8 spatial positions, then pool along W via pair averaging.
    # Output row length = Wp = W/2. We'll accumulate full-W normalized values, sum them across 8 positions,
    # then pair-average along W at the end.
    acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    d0 = dp * 2
    h0 = hp * 2

    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            d = d0 + dd
            h = h0 + hh
            row_off = base + d * (H * W) + h * W + w_offs
            x = tl.load(x_ptr + row_off, mask=w_mask, other=0.0).to(tl.float32)
            x = x + sum_w
            # LayerNorm over W
            x_m = tl.where(w_mask, x, 0.0)
            mean = tl.sum(x_m, axis=0) * inv_W
            diff = tl.where(w_mask, x - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) * inv_W
            rstd = 1.0 / tl.sqrt(var + eps)
            normed = (x - mean) * rstd * gamma + beta
            # Add for both copies (we will add for both d and h positions => factor of 4 with 2x2)
            acc += normed

    # acc has sum of 4 normalized rows (over 2 d × 2 h). For AvgPool 2x2x2, we also need to
    # average along W in pairs: out[w'] = (acc[2w'] + acc[2w'+1]) / 8
    # We use shift trick: pair sum via masking.
    # Compute pair sums: for even index w'*2, take acc[w] + acc[w+1].
    # Easiest: load acc into two halves using tl.arange.
    wp_offs = tl.arange(0, BLOCK_W // 2)
    wp_mask = wp_offs < Wp

    # Use tl.where to gather even and odd:
    # Create even / odd masks
    even = (w_offs % 2) == 0
    odd = (w_offs % 2) == 1
    # acc shifted: we use the trick of summing pairs by reshaping. Since BLOCK_W is constexpr,
    # we can do: pairs computed via masked sums won't easily reshape. Instead, just store full acc
    # to a temp via tl arithmetic... Use the fact that BLOCK_W is power-of-two constexpr:
    # split into even and odd by selecting with masks and shifting.
    # Method: use tl.reshape (Triton supports reshape on constexpr shapes).
    acc2 = tl.reshape(acc, (BLOCK_W // 2, 2))
    pair_sum = tl.sum(acc2, axis=1)  # shape [BLOCK_W//2]

    pooled = pair_sum * (1.0 / 8.0)

    # GELU
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

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
        self.norm_shape = tuple(norm_shape) if isinstance(norm_shape, (list, tuple)) else (norm_shape,)

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape

        # Check if we can apply our fused kernel: LayerNorm normalizes over the last
        # len(norm_shape) dims of x. norm_shape=(C_norm,) means it normalizes over W only.
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

        # Fallback
        x = x + self.sum_weight
        x = self.norm(x)
        x = self.avg_pool(x)
        x = self.gelu(x)
        return x