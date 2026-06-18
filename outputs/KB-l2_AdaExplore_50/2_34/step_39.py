import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROWS': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_ROWS': 2}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_ROWS': 4}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_ROWS': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_ROWS': 8}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_ROWS': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_ROWS': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_ROWS': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_ROWS': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'C'],
)
@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, C,
    eps, scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_ROWS
    rows = row_start + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < M

    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < C
    mask = row_mask[:, None] & col_mask[None, :]

    offs = rows[:, None] * C + cols[None, :]
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    inv_C = 1.0 / C
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=1)
    mean = sum_x * inv_C
    xc = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) * inv_C
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)

    y = xc * rstd[:, None] * g[None, :] + b[None, :]
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = gelu * scaling_factor

    tl.store(out_ptr + offs, out, mask=mask)


def ln_gelu_scale(x_channel_last, gamma, beta, eps, scaling_factor):
    # x_channel_last: (..., C) contiguous
    C = x_channel_last.shape[-1]
    M = x_channel_last.numel() // C
    out = torch.empty_like(x_channel_last)
    BLOCK_C = triton.next_power_of_2(C)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_ROWS']),)
    _ln_gelu_scale_kernel[grid](
        x_channel_last, gamma, beta, out,
        M, C, eps, scaling_factor,
        BLOCK_C=BLOCK_C,
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
        # x: (N, C, D', H', W'). LayerNorm normalizes over the last dim W'
        # (which equals out_channels in this configuration).
        x = x.contiguous()
        out = ln_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias, self.eps, self.scaling_factor)
        return out