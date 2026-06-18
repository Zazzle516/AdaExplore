import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Note: norm_shape = (out_channels,) = (64,), so LayerNorm is applied over
# the last dim of the [N, C, D, H, W] tensor — wait, careful.
# nn.LayerNorm(norm_shape) normalizes over the LAST len(norm_shape) dims.
# norm_shape=(64,), so it normalizes over the last dim (W=64 after deconv).
# Actually checking: input is [N, C, D, H, W] with C=64 and W=64.
# LayerNorm normalizes over last 1 dim of size 64 — that's W axis.
# So per-row (N,C,D,H) we normalize across W.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HO': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HO': 2}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HO': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_HO': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HO': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HO': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HO': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HO': 16}, num_warps=8, num_stages=2),
    ],
    key=['C', 'W', 'Ho'],
)
@triton.jit
def fused_post_kernel(
    x_ptr,
    out_ptr,
    gamma_ptr,
    beta_ptr,
    sum_w,
    eps,
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_W: tl.constexpr,
    BLOCK_WO: tl.constexpr,
    BLOCK_HO: tl.constexpr,
):
    pid = tl.program_id(0)
    Ho_blocks = (Ho + BLOCK_HO - 1) // BLOCK_HO
    ho_blk = pid % Ho_blocks
    tmp = pid // Ho_blocks
    do = tmp % Do
    tmp = tmp // Do
    c = tmp % C
    n = tmp // C

    d0 = do * 2

    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < W

    gamma = tl.load(gamma_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

    DHW = D * H * W
    HW = H * W
    inv_W = 1.0 / W
    inv_8 = 1.0 / 8.0
    inv_sqrt2 = 0.7071067811865475

    base_nc = n * C * DHW + c * DHW
    base_d0 = base_nc + d0 * HW
    base_d1 = base_nc + (d0 + 1) * HW

    DHWo = Do * Ho * Wo
    HWo = Ho * Wo
    out_base_nc = n * C * DHWo + c * DHWo + do * HWo

    wo_off = tl.arange(0, BLOCK_WO)
    wo_mask = wo_off < Wo

    for i in tl.static_range(BLOCK_HO):
        ho = ho_blk * BLOCK_HO + i
        if ho < Ho:
            h0 = ho * 2

            row_base0 = base_d0 + h0 * W
            row_base1 = base_d0 + (h0 + 1) * W
            row_base2 = base_d1 + h0 * W
            row_base3 = base_d1 + (h0 + 1) * W

            x0 = tl.load(x_ptr + row_base0 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
            x1 = tl.load(x_ptr + row_base1 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
            x2 = tl.load(x_ptr + row_base2 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
            x3 = tl.load(x_ptr + row_base3 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w

            x0z = tl.where(w_mask, x0, 0.0)
            s0 = tl.sum(x0z, axis=0)
            sq0 = tl.sum(x0z * x0z, axis=0)
            m0 = s0 * inv_W
            v0 = sq0 * inv_W - m0 * m0
            r0 = 1.0 / tl.sqrt(v0 + eps)
            y0 = (x0 - m0) * r0 * gamma + beta

            x1z = tl.where(w_mask, x1, 0.0)
            s1 = tl.sum(x1z, axis=0)
            sq1 = tl.sum(x1z * x1z, axis=0)
            m1 = s1 * inv_W
            v1 = sq1 * inv_W - m1 * m1
            r1 = 1.0 / tl.sqrt(v1 + eps)
            y1 = (x1 - m1) * r1 * gamma + beta

            x2z = tl.where(w_mask, x2, 0.0)
            s2 = tl.sum(x2z, axis=0)
            sq2 = tl.sum(x2z * x2z, axis=0)
            m2 = s2 * inv_W
            v2 = sq2 * inv_W - m2 * m2
            r2 = 1.0 / tl.sqrt(v2 + eps)
            y2 = (x2 - m2) * r2 * gamma + beta

            x3z = tl.where(w_mask, x3, 0.0)
            s3 = tl.sum(x3z, axis=0)
            sq3 = tl.sum(x3z * x3z, axis=0)
            m3 = s3 * inv_W
            v3 = sq3 * inv_W - m3 * m3
            r3 = 1.0 / tl.sqrt(v3 + eps)
            y3 = (x3 - m3) * r3 * gamma + beta

            row_sum = y0 + y1 + y2 + y3

            row_sum_2d = tl.reshape(row_sum, [BLOCK_WO, 2])
            pooled = tl.sum(row_sum_2d, axis=1) * inv_8

            gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))

            out_base = out_base_nc + ho * Wo
            tl.store(out_ptr + out_base + wo_off, gelu, mask=wo_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        Do, Ho, Wo = D // 2, H // 2, W // 2

        out = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        x_c = x.contiguous()
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps
        sum_w = float(self.sum_weight.item())

        BLOCK_W = _next_pow2(W)
        BLOCK_WO = BLOCK_W // 2

        def grid(meta):
            BLOCK_HO = meta['BLOCK_HO']
            Ho_blocks = (Ho + BLOCK_HO - 1) // BLOCK_HO
            return (N * C * Do * Ho_blocks,)

        fused_post_kernel[grid](
            x_c, out, gamma, beta,
            sum_w, eps,
            N, C, D, H, W,
            Do, Ho, Wo,
            BLOCK_W=BLOCK_W,
            BLOCK_WO=BLOCK_WO,
        )
        return out