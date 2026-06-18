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
    # one program per (n, h, w_tile); reduces over C for a tile of W
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)
    HW = H * W
    n = pid // H
    h = pid % H

    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = offs_w < W

    base = n * C * HW + h * W
    # offsets: c*HW + w
    offs = offs_c[:, None] * HW + offs_w[None, :]  # [BLOCK_C, BLOCK_W]
    mask = c_mask[:, None] & w_mask[None, :]
    x = tl.load(x_ptr + base + offs, mask=mask, other=float('inf'))
    m = tl.min(x, axis=0)  # [BLOCK_W]
    tl.store(out_ptr + n * HW + h * W + offs_w, m, mask=w_mask)


@triton.jit
def sum_gelu_bias_kernel(
    x_ptr, out_ptr, bias_ptr,
    N, H, W,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # x: (N, 1, H, W) -> sum along H -> (N, 1, 1, W), then gelu + bias
    # one program per (n, w_tile)
    pid = tl.program_id(0)
    pid_w = tl.program_id(1)
    n = pid

    offs_h = tl.arange(0, BLOCK_H)
    h_mask = offs_h < H
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = offs_w < W

    base = n * H * W
    offs = offs_h[:, None] * W + offs_w[None, :]
    mask = h_mask[:, None] & w_mask[None, :]
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    s = tl.sum(x, axis=0)  # [BLOCK_W]

    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * s * (1.0 + tl.erf(s * inv_sqrt2))

    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + n * W + offs_w, out, mask=w_mask)


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

        # min along channels (tiled over W)
        min_out = torch.empty((N, 1, H, W), device=x.device, dtype=x.dtype)
        BLOCK_C = _next_pow2(C)
        BLOCK_W = 64 if W >= 64 else _next_pow2(W)
        grid = (N * H, triton.cdiv(W, BLOCK_W))
        min_reduce_kernel[grid](x, min_out, N, C, H, W, BLOCK_C=BLOCK_C, BLOCK_W=BLOCK_W, num_warps=4)

        # sum along H, gelu, + bias (tiled over W)
        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)
        BLOCK_H = _next_pow2(H)
        BLOCK_W2 = 64 if W >= 64 else _next_pow2(W)
        grid2 = (N, triton.cdiv(W, BLOCK_W2))
        sum_gelu_bias_kernel[grid2](min_out, out, self.bias, N, H, W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W2, num_warps=4)

        return out