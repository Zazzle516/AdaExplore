import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_W': 64}, num_warps=8, num_stages=2),
    ],
    key=['C', 'H', 'W'],
)
@triton.jit
def min_sum_gelu_bias_kernel(
    x_ptr,         # input: NCHW contiguous, [N, C, H, W]
    out_ptr,       # output: [N, W]
    bias_ptr,      # bias scalar (1,1,1)
    N, C, H, W,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    w_blocks = tl.cdiv(W, BLOCK_W)
    n = pid // w_blocks
    wb = pid % w_blocks

    w_offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)  # [BW]
    w_mask = w_offs < W

    c_offs = tl.arange(0, BLOCK_C)  # [BC]
    c_mask = c_offs < C

    INF = float('inf')

    sum_val = tl.zeros([BLOCK_W], dtype=tl.float32)

    n_base = n * C * H * W
    HW = H * W

    for h in range(0, H):
        # x[n, c, h, w] -> offset = n_base + c*HW + h*W + w
        ptrs = x_ptr + n_base + c_offs[None, :] * HW + h * W + w_offs[:, None]
        mask = w_mask[:, None] & c_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=INF)
        mn = tl.min(vals, axis=1)  # [BW]
        sum_val = sum_val + mn

    g = 0.5 * sum_val * (1.0 + tl.erf(sum_val * 0.70710678118654752440))
    b = tl.load(bias_ptr)
    res = g + b

    out_offs = n * W + w_offs
    tl.store(out_ptr + out_offs, res, mask=w_mask)


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

        grid = lambda meta: (N * triton.cdiv(W, meta['BLOCK_W']),)
        min_sum_gelu_bias_kernel[grid](
            x, out, self.bias,
            N, C, H, W,
            BLOCK_C=BLOCK_C,
        )
        return out