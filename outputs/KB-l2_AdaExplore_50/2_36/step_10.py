import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def min_reduce_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, h, w)
    pid = tl.program_id(0)
    HW = H * W
    n = pid // HW
    rem = pid % HW
    h = rem // W
    w = rem % W

    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C
    base = n * C * HW + h * W + w
    x = tl.load(x_ptr + base + offs_c * HW, mask=mask, other=float('inf'))
    m = tl.min(x, axis=0)
    tl.store(out_ptr + n * HW + h * W + w, m)


@triton.jit
def sum_gelu_bias_kernel(
    x_ptr, out_ptr, bias_ptr,
    N, H, W,
    BLOCK_H: tl.constexpr,
):
    # x: (N, 1, H, W) -> sum along H -> (N, 1, 1, W), then gelu + bias
    # one program per (n, w)
    pid = tl.program_id(0)
    n = pid // W
    w = pid % W

    offs_h = tl.arange(0, BLOCK_H)
    mask = offs_h < H
    base = n * H * W + w
    x = tl.load(x_ptr + base + offs_h * W, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)

    # GELU exact: 0.5 * x * (1 + erf(x / sqrt(2)))
    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * s * (1.0 + tl.erf(s * inv_sqrt2))

    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + n * W + w, out)


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, H, W = x.shape
        x = x.contiguous()

        # min along channels
        min_out = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)
        BLOCK_C = _next_pow2(C)
        grid = (N * H * W,)
        min_reduce_kernel[grid](x, min_out, N, C, H, W, BLOCK_C=BLOCK_C)

        # sum along H, gelu, + bias
        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)
        BLOCK_H = _next_pow2(H)
        grid2 = (N * W,)
        sum_gelu_bias_kernel[grid2](min_out, out, self.bias, N, H, W, BLOCK_H=BLOCK_H)

        return out