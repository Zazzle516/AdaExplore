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
    weight_ptr,      # LayerNorm weight [C]
    bias_ptr,        # LayerNorm bias [C]
    sum_weight,      # scalar
    eps,
    N, C, D, H, W,
    Dp, Hp, Wp,      # pooled dims
    BLOCK_C: tl.constexpr,
):
    # one program per (n, dp, hp, wp)
    pid = tl.program_id(0)
    # decompose pid -> n, dp, hp, wp
    wp = pid % Wp
    tmp = pid // Wp
    hp = tmp % Hp
    tmp = tmp // Hp
    dp = tmp % Dp
    n = tmp // Dp

    # base coords in input
    d0 = dp * 2
    h0 = hp * 2
    w0 = wp * 2

    # We need to compute LayerNorm over C for each of 8 spatial positions,
    # then average them, then apply GELU.

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # Accumulator for the average over the 2x2x2 window
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    # iterate 8 positions
    for dd in tl.static_range(0, 2):
        for hh in tl.static_range(0, 2):
            for ww in tl.static_range(0, 2):
                d = d0 + dd
                h = h0 + hh
                w = w0 + ww
                # offset: ((n*C + c)*D + d)*H*W + h*W + w
                # we vary c in block
                base = ((n * C + 0) * D + d) * H * W + h * W + w
                stride_c = D * H * W
                ptrs = x_ptr + base + c_offs * stride_c
                vals = tl.load(ptrs, mask=c_mask, other=0.0)
                vals = vals + sum_weight
                # layernorm over C
                vals_f = vals.to(tl.float32)
                # mean
                sum_v = tl.sum(tl.where(c_mask, vals_f, 0.0), axis=0)
                mean = sum_v / C
                diff = tl.where(c_mask, vals_f - mean, 0.0)
                var = tl.sum(diff * diff, axis=0) / C
                inv = 1.0 / tl.sqrt(var + eps)
                normed = diff * inv
                w_ln = tl.load(weight_ptr + c_offs, mask=c_mask, other=0.0)
                b_ln = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0)
                normed = normed * w_ln + b_ln
                acc = acc + normed

    acc = acc / 8.0
    # GELU (exact using erf)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    # store to out: [N, C, Dp, Hp, Wp]
    out_base = ((n * C + 0) * Dp + dp) * Hp * Wp + hp * Wp + wp
    out_stride_c = Dp * Hp * Wp
    out_ptrs = out_ptr + out_base + c_offs * out_stride_c
    tl.store(out_ptrs, gelu, mask=c_mask)


def fused_post(x, weight, bias, sum_weight, eps):
    N, C, D, H, W = x.shape
    Dp, Hp, Wp = D // 2, H // 2, W // 2
    out = torch.empty((N, C, Dp, Hp, Wp), dtype=x.dtype, device=x.device)
    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * Dp * Hp * Wp,)
    fused_post_kernel[grid](
        x, out, weight, bias,
        float(sum_weight), float(eps),
        N, C, D, H, W, Dp, Hp, Wp,
        BLOCK_C=BLOCK_C,
        num_warps=2,
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
        # Use fused kernel only when norm is over channel dim and pool is 2x2x2
        if (isinstance(self.pool_kernel_size, tuple) and self.pool_kernel_size == (2, 2, 2)
                and self.norm.normalized_shape == (x.shape[1],)):
            return fused_post(
                x,
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