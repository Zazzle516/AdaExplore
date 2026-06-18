import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
    ],
    key=['C', 'W'],
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
):
    pid = tl.program_id(0)
    ho = pid % Ho
    tmp = pid // Ho
    do = tmp % Do
    tmp = tmp // Do
    c = tmp % C
    n = tmp // C

    d0 = do * 2
    h0 = ho * 2

    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < W

    gamma = tl.load(gamma_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

    DHW = D * H * W
    HW = H * W
    inv_W = 1.0 / W

    base_nc = n * C * DHW + c * DHW
    row0_base = base_nc + d0 * HW + h0 * W
    row1_base = base_nc + d0 * HW + (h0 + 1) * W
    row2_base = base_nc + (d0 + 1) * HW + h0 * W
    row3_base = base_nc + (d0 + 1) * HW + (h0 + 1) * W

    # Load all 4 rows up front to enable better pipelining
    x0 = tl.load(x_ptr + row0_base + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    x1 = tl.load(x_ptr + row1_base + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    x2 = tl.load(x_ptr + row2_base + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    x3 = tl.load(x_ptr + row3_base + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w

    # Per-row LayerNorm (over W axis)
    x0z = tl.where(w_mask, x0, 0.0)
    m0 = tl.sum(x0z, axis=0) * inv_W
    d0v = tl.where(w_mask, x0 - m0, 0.0)
    v0 = tl.sum(d0v * d0v, axis=0) * inv_W
    r0 = 1.0 / tl.sqrt(v0 + eps)
    y0 = (x0 - m0) * r0 * gamma + beta

    x1z = tl.where(w_mask, x1, 0.0)
    m1 = tl.sum(x1z, axis=0) * inv_W
    d1v = tl.where(w_mask, x1 - m1, 0.0)
    v1 = tl.sum(d1v * d1v, axis=0) * inv_W
    r1 = 1.0 / tl.sqrt(v1 + eps)
    y1 = (x1 - m1) * r1 * gamma + beta

    x2z = tl.where(w_mask, x2, 0.0)
    m2 = tl.sum(x2z, axis=0) * inv_W
    d2v = tl.where(w_mask, x2 - m2, 0.0)
    v2 = tl.sum(d2v * d2v, axis=0) * inv_W
    r2 = 1.0 / tl.sqrt(v2 + eps)
    y2 = (x2 - m2) * r2 * gamma + beta

    x3z = tl.where(w_mask, x3, 0.0)
    m3 = tl.sum(x3z, axis=0) * inv_W
    d3v = tl.where(w_mask, x3 - m3, 0.0)
    v3 = tl.sum(d3v * d3v, axis=0) * inv_W
    r3 = 1.0 / tl.sqrt(v3 + eps)
    y3 = (x3 - m3) * r3 * gamma + beta

    row_sum = y0 + y1 + y2 + y3

    row_sum_2d = tl.reshape(row_sum, [BLOCK_WO, 2])
    pooled = tl.sum(row_sum_2d, axis=1) * (1.0 / 8.0)

    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * pooled * (1.0 + tl.math.erf(pooled * inv_sqrt2))

    wo_off = tl.arange(0, BLOCK_WO)
    wo_mask = wo_off < Wo
    DHWo = Do * Ho * Wo
    HWo = Ho * Wo
    out_base = n * C * DHWo + c * DHWo + do * HWo + ho * Wo
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
        pk = self.pool_kernel_size
        assert pk == (2, 2, 2) or list(pk) == [2, 2, 2]
        Do, Ho, Wo = D // 2, H // 2, W // 2

        out = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        x_c = x.contiguous()
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps
        sum_w = float(self.sum_weight.item())

        BLOCK_W = _next_pow2(W)
        BLOCK_WO = BLOCK_W // 2
        assert BLOCK_WO >= Wo, "BLOCK_WO must cover Wo"
        grid = (N * C * Do * Ho,)
        fused_post_kernel[grid](
            x_c, out, gamma, beta,
            sum_w, eps,
            N, C, D, H, W,
            Do, Ho, Wo,
            BLOCK_W=BLOCK_W,
            BLOCK_WO=BLOCK_WO,
        )
        return out