import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def min_sum_gelu_bias_kernel(
    x_ptr,         # input: [N, C, H, W]
    out_ptr,       # output: [N, 1, 1, W]
    bias_ptr,      # bias scalar
    N, C, H, W,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    # one program per (n, w)
    pid = tl.program_id(0)
    n = pid // W
    w = pid % W

    INF = float('inf')
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    h_offs = tl.arange(0, BLOCK_H)

    base = n * stride_n + w * stride_w

    sum_val = tl.zeros([], dtype=tl.float32)

    for h_start in range(0, H, BLOCK_H):
        h_cur = h_start + h_offs
        h_mask = h_cur < H
        # ptrs shape [BLOCK_H, BLOCK_C]
        ptrs = base + h_cur[:, None] * stride_h + c_offs[None, :] * stride_c
        mask = h_mask[:, None] & c_mask[None, :]
        vals = tl.load(x_ptr + ptrs, mask=mask, other=INF)
        # min over channel axis -> [BLOCK_H]
        mn = tl.min(vals, axis=1)
        # mask out invalid h with 0
        mn = tl.where(h_mask, mn, 0.0)
        sum_val += tl.sum(mn, axis=0)

    # GELU (exact)
    g = 0.5 * sum_val * (1.0 + tl.erf(sum_val * 0.70710678118654752440))
    b = tl.load(bias_ptr)
    res = g + b

    out_ptr_off = n * W + w
    tl.store(out_ptr + out_ptr_off, res)


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

        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)

        BLOCK_C = triton.next_power_of_2(C)
        if BLOCK_C < 16:
            BLOCK_C = 16

        BLOCK_H = 16

        grid = (N * W,)
        min_sum_gelu_bias_kernel[grid](
            x, out, self.bias,
            N, C, H, W,
            C * H * W, H * W, W, 1,
            BLOCK_C=BLOCK_C,
            BLOCK_H=BLOCK_H,
            num_warps=4,
        )
        return out