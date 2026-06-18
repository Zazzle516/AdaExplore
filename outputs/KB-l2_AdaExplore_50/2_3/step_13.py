import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# LayerNorm here is over the last-dim of size out_channels=64 (the C dim after permutation logic)
# But actually norm_shape is (out_channels,) = (64,) and applied to last dim of x which is W=64.
# Wait: x shape after conv is [N, C=64, D=32, H=64, W=64], LN normalizes over last len(norm_shape) dims.
# norm_shape=(64,) so it normalizes over last dim W=64. That matches kernel 0.

# Strategy: one program per (n, od, oh, ow_tile) computing a tile of OW outputs across all C.
# But LN is over W axis, so we need full W row to compute mean/var for each (n,c,id,ih).
# Better: one program per (n, c, od, oh) - compute LN for 4 input rows (each of length W),
# then pool 2x2x2 -> OW outputs. Same as kernel 0 but optimize.
#
# Optimization vs kernel 0: process multiple (od, oh) per program? Or process multiple c?
# Actually kernel 0 grid is N*C*OD*OH = 32*64*16*32 = 1,048,576 programs. That's a lot.
# Each program does ~4 * W = 256 fp32 work for LN + tiny pool. Mostly memory bound.
# 
# Let's keep similar structure but make sure we handle things efficiently.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 128}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_W': 256}, num_warps=4, num_stages=2),
    ],
    key=['W'],
)
@triton.jit
def fused_post_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    sum_weight,
    eps,
    inv_W,
    N, C, D, H, W,
    OD, OH, OW,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    oh = pid % OH
    tmp = pid // OH
    od = tmp % OD
    tmp = tmp // OD
    c = tmp % C
    n = tmp // C

    w_idx = tl.arange(0, BLOCK_W)
    w_mask = w_idx < W

    gamma = tl.load(gamma_ptr + w_idx, mask=w_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + w_idx, mask=w_mask, other=0.0).to(tl.float32)

    base_nc = (n * C + c) * D * H * W

    id0 = 2 * od
    id1 = 2 * od + 1
    ih0 = 2 * oh
    ih1 = 2 * oh + 1

    # Load 4 rows
    p00 = x_ptr + base_nc + (id0 * H + ih0) * W + w_idx
    p01 = x_ptr + base_nc + (id0 * H + ih1) * W + w_idx
    p10 = x_ptr + base_nc + (id1 * H + ih0) * W + w_idx
    p11 = x_ptr + base_nc + (id1 * H + ih1) * W + w_idx

    x00 = tl.load(p00, mask=w_mask, other=0.0).to(tl.float32) + sum_weight
    x01 = tl.load(p01, mask=w_mask, other=0.0).to(tl.float32) + sum_weight
    x10 = tl.load(p10, mask=w_mask, other=0.0).to(tl.float32) + sum_weight
    x11 = tl.load(p11, mask=w_mask, other=0.0).to(tl.float32) + sum_weight

    # LN per row
    def ln(x):
        x_safe = tl.where(w_mask, x, 0.0)
        mean = tl.sum(x_safe, axis=0) * inv_W
        diff = tl.where(w_mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_W
        rstd = 1.0 / tl.sqrt(var + eps)
        return (x - mean) * rstd * gamma + beta

    # inline ln to avoid potential issues
    x_safe = tl.where(w_mask, x00, 0.0)
    mean = tl.sum(x_safe, axis=0) * inv_W
    diff = tl.where(w_mask, x00 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) * inv_W
    rstd = 1.0 / tl.sqrt(var + eps)
    n00 = (x00 - mean) * rstd * gamma + beta

    x_safe = tl.where(w_mask, x01, 0.0)
    mean = tl.sum(x_safe, axis=0) * inv_W
    diff = tl.where(w_mask, x01 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) * inv_W
    rstd = 1.0 / tl.sqrt(var + eps)
    n01 = (x01 - mean) * rstd * gamma + beta

    x_safe = tl.where(w_mask, x10, 0.0)
    mean = tl.sum(x_safe, axis=0) * inv_W
    diff = tl.where(w_mask, x10 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) * inv_W
    rstd = 1.0 / tl.sqrt(var + eps)
    n10 = (x10 - mean) * rstd * gamma + beta

    x_safe = tl.where(w_mask, x11, 0.0)
    mean = tl.sum(x_safe, axis=0) * inv_W
    diff = tl.where(w_mask, x11 - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) * inv_W
    rstd = 1.0 / tl.sqrt(var + eps)
    n11 = (x11 - mean) * rstd * gamma + beta

    acc = n00 + n01 + n10 + n11

    acc2 = tl.reshape(acc, (BLOCK_W // 2, 2))
    pair_sum = tl.sum(acc2, axis=1)

    pooled = pair_sum * 0.125

    k0 = 0.7978845608028654
    k1 = 0.044715
    x3 = pooled * pooled * pooled
    inner = k0 * (pooled + k1 * x3)
    two_y = 2.0 * inner
    sig = 1.0 / (1.0 + tl.exp(-two_y))
    tanh_v = 2.0 * sig - 1.0
    gelu = 0.5 * pooled * (1.0 + tanh_v)

    ow_idx = tl.arange(0, BLOCK_W // 2)
    ow_mask = ow_idx < OW
    out_base = ((n * C + c) * OD + od) * OH * OW + oh * OW
    out_ptrs = out_ptr + out_base + ow_idx
    tl.store(out_ptrs, gelu, mask=ow_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.out_channels = out_channels
        self.pool_kernel_size = pool_kernel_size
        self.norm_shape = tuple(norm_shape) if not isinstance(norm_shape, int) else (norm_shape,)

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        pk = self.pool_kernel_size

        ns = self.norm_shape
        fast = (
            pk == (2, 2, 2)
            and len(ns) == 1
            and ns[0] == W
            and (D % 2 == 0) and (H % 2 == 0) and (W % 2 == 0)
        )
        if not fast:
            x = x + self.sum_weight
            x = self.norm(x)
            x = self.avg_pool(x)
            x = self.gelu(x)
            return x

        OD = D // 2
        OH = H // 2
        OW = W // 2

        x = x.contiguous()
        out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)

        total = N * C * OD * OH
        grid = (total,)

        fused_post_kernel[grid](
            x, out,
            self.norm.weight, self.norm.bias,
            float(self.sum_weight.item()),
            float(self.norm.eps),
            1.0 / W,
            N, C, D, H, W,
            OD, OH, OW,
        )
        return out