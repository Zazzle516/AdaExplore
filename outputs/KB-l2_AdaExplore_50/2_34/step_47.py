import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'ROWS_PER_PROG': 1}, num_warps=2, num_stages=2),
        triton.Config({'ROWS_PER_PROG': 4}, num_warps=4, num_stages=2),
        triton.Config({'ROWS_PER_PROG': 8}, num_warps=4, num_stages=2),
        triton.Config({'ROWS_PER_PROG': 8}, num_warps=8, num_stages=2),
        triton.Config({'ROWS_PER_PROG': 16}, num_warps=4, num_stages=2),
        triton.Config({'ROWS_PER_PROG': 16}, num_warps=8, num_stages=2),
    ],
    key=['C'],
)
@triton.jit
def layernorm_gelu_scale_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    M, C,
    eps, scaling_factor,
    BLOCK_C: tl.constexpr,
    ROWS_PER_PROG: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    inv_sqrt2 = 0.7071067811865475

    row_start = pid * ROWS_PER_PROG
    for i in tl.static_range(ROWS_PER_PROG):
        row = row_start + i
        row_valid = row < M
        x_row_ptr = x_ptr + row * C
        out_row_ptr = out_ptr + row * C

        x = tl.load(x_row_ptr + offs, mask=mask & row_valid, other=0.0).to(tl.float32)

        sum_x = tl.sum(x, axis=0)
        sum_x2 = tl.sum(x * x, axis=0)
        mean = sum_x / C
        var = sum_x2 / C - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        y = (x - mean) * rstd * w + b
        gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
        gelu = gelu * scaling_factor

        tl.store(out_row_ptr + offs, gelu, mask=mask & row_valid)


def layernorm_gelu_scale(x, weight, bias, eps, scaling_factor):
    x_contig = x.contiguous()
    C_norm = weight.shape[0]
    M = x_contig.numel() // C_norm
    out = torch.empty_like(x_contig)

    BLOCK_C = triton.next_power_of_2(C_norm)
    grid = lambda meta: (triton.cdiv(M, meta['ROWS_PER_PROG']),)
    layernorm_gelu_scale_kernel[grid](
        x_contig, out, weight, bias,
        M, C_norm,
        eps, scaling_factor,
        BLOCK_C=BLOCK_C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = layernorm_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias, self.eps, self.scaling_factor)
        return x