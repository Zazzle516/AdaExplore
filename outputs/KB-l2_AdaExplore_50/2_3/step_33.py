import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=3),
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

    row_sum = tl.zeros([BLOCK_W], dtype=tl.float32)

    for i in tl.static_range(0, 4):
        dd = d0 + (i // 2)
        hh = h0 + (i % 2)
        row_base = base_nc + dd * HW + hh * W
        x = tl.load(x_ptr + row_base + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w

        x_zero = tl.where(w_mask, x, 0.0)
        mean = tl.sum(x_zero, axis=0) * inv_W
        diff = tl.where(w_mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_W
        rstd = 1.0 / tl.sqrt(var + eps)
        y = (x - mean) * rstd * gamma + beta
        row_sum = row_sum + y

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