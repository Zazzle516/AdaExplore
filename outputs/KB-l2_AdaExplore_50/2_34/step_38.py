import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'ROWS_PER_PROG': 1}, num_warps=2),
        triton.Config({'ROWS_PER_PROG': 1}, num_warps=4),
        triton.Config({'ROWS_PER_PROG': 2}, num_warps=2),
        triton.Config({'ROWS_PER_PROG': 2}, num_warps=4),
        triton.Config({'ROWS_PER_PROG': 4}, num_warps=2),
        triton.Config({'ROWS_PER_PROG': 4}, num_warps=4),
        triton.Config({'ROWS_PER_PROG': 8}, num_warps=4),
        triton.Config({'ROWS_PER_PROG': 8}, num_warps=8),
        triton.Config({'ROWS_PER_PROG': 16}, num_warps=4),
        triton.Config({'ROWS_PER_PROG': 16}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, C,
    eps, scaling_factor,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    cols = tl.arange(0, BLOCK_C)
    mask_c = cols < C

    g = tl.load(gamma_ptr + cols, mask=mask_c, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + cols, mask=mask_c, other=0.0).to(tl.float32)

    inv_sqrt2 = 0.7071067811865475
    inv_C = 1.0 / C

    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        if row < M:
            base = row * C
            x = tl.load(x_ptr + base + cols, mask=mask_c, other=0.0).to(tl.float32)

            sum_x = tl.sum(tl.where(mask_c, x, 0.0), axis=0)
            mean = sum_x * inv_C
            xc = tl.where(mask_c, x - mean, 0.0)
            var = tl.sum(xc * xc, axis=0) * inv_C
            rstd = 1.0 / tl.sqrt(var + eps)

            y = xc * rstd * g + b
            gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
            out = gelu * scaling_factor

            tl.store(out_ptr + base + cols, out, mask=mask_c)


def ln_gelu_scale(x_channel_last, gamma, beta, eps, scaling_factor):
    C = x_channel_last.shape[-1]
    M = x_channel_last.numel() // C
    out = torch.empty_like(x_channel_last)
    BLOCK_C = triton.next_power_of_2(C)

    grid = lambda meta: ((M + meta['ROWS_PER_PROG'] - 1) // meta['ROWS_PER_PROG'],)
    _ln_gelu_scale_kernel[grid](
        x_channel_last, gamma, beta, out,
        M, C, eps, scaling_factor,
        BLOCK_C=BLOCK_C,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        out = ln_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias, self.eps, self.scaling_factor)
        return out