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
    BLOCK_W: tl.constexpr,
):
    # one program per (n, h, w_tile)
    pid = tl.program_id(0)
    HW = H * W
    n_h = pid // ((W + BLOCK_W - 1) // BLOCK_W)
    w_tile = pid % ((W + BLOCK_W - 1) // BLOCK_W)
    n = n_h // H
    h = n_h % H

    offs_c = tl.arange(0, BLOCK_C)
    offs_w = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_c = offs_c < C
    mask_w = offs_w < W

    base = n * C * HW + h * W
    # x[c, w] = x_ptr[base + c*HW + w]
    ptrs = base + offs_c[:, None] * HW + offs_w[None, :]
    mask = mask_c[:, None] & mask_w[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=float('inf'))
    m = tl.min(x, axis=0)
    tl.store(out_ptr + n * HW + h * W + offs_w, m, mask=mask_w)


@triton.jit
def sum_gelu_bias_kernel(
    x_ptr, out_ptr, bias_ptr,
    N, H, W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, w_tile)
    pid = tl.program_id(0)
    n_tiles_w = (W + BLOCK_W - 1) // BLOCK_W
    n = pid // n_tiles_w
    w_tile = pid % n_tiles_w

    offs_h = tl.arange(0, BLOCK_H)
    offs_w = w_tile * BLOCK_W + tl.arange(0, BLOCK_W)
    mask_h = offs_h < H
    mask_w = offs_w < W

    base = n * H * W
    ptrs = base + offs_h[:, None] * W + offs_w[None, :]
    mask = mask_h[:, None] & mask_w[None, :]
    x = tl.load(x_ptr + ptrs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)

    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * s * (1.0 + tl.erf(s * inv_sqrt2))

    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + n * W + offs_w, out, mask=mask_w)


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

        BLOCK_C = _next_pow2(C)
        BLOCK_W = 64
        n_tiles_w = (W + BLOCK_W - 1) // BLOCK_W

        min_out = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)
        grid = (N * H * n_tiles_w,)
        min_reduce_kernel[grid](x, min_out, N, C, H, W, BLOCK_C=BLOCK_C, BLOCK_W=BLOCK_W, num_warps=4)

        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)
        BLOCK_H = _next_pow2(H)
        BLOCK_W2 = 64
        n_tiles_w2 = (W + BLOCK_W2 - 1) // BLOCK_W2
        grid2 = (N * n_tiles_w2,)
        sum_gelu_bias_kernel[grid2](min_out, out, self.bias, N, H, W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W2, num_warps=4)

        return out