import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def min_sum_gelu_bias_kernel_nhwc(
    x_ptr,         # input: NHWC contiguous, [N, H, W, C]
    out_ptr,       # output: [N, W] (then viewed as [N,1,1,W])
    bias_ptr,      # bias scalar (1,1,1)
    N, C, H, W,
    BLOCK_C: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, w_block)
    pid = tl.program_id(0)
    w_blocks = tl.cdiv(W, BLOCK_W)
    n = pid // w_blocks
    wb = pid % w_blocks

    w_offs = wb * BLOCK_W + tl.arange(0, BLOCK_W)  # [BW]
    w_mask = w_offs < W

    c_offs = tl.arange(0, BLOCK_C)  # [BC]
    c_mask = c_offs < C

    INF = float('inf')

    # accumulator [BW]
    sum_val = tl.zeros([BLOCK_W], dtype=tl.float32)

    # base pointer for x[n, 0, 0, 0]
    n_base = n * H * W * C

    for h in range(0, H):
        # x[n, h, w_offs, c_offs] -> shape [BW, BC]
        # offset = n_base + h*W*C + w_offs[:,None]*C + c_offs[None,:]
        ptrs = x_ptr + n_base + h * W * C + w_offs[:, None] * C + c_offs[None, :]
        mask = w_mask[:, None] & c_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=INF)
        mn = tl.min(vals, axis=1)  # [BW]
        sum_val = sum_val + mn

    # GELU (exact)
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
        # Use channels_last for the conv to get NHWC contiguous output
        x = x.to(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        N, C, H, W = x.shape

        # ensure NHWC contiguous layout
        # x has channels_last memory format; reinterpret as [N, H, W, C] contiguous
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()

        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)

        BLOCK_C = triton.next_power_of_2(C)
        if BLOCK_C < 16:
            BLOCK_C = 16
        BLOCK_W = 16

        grid = ((N * triton.cdiv(W, BLOCK_W)),)
        min_sum_gelu_bias_kernel_nhwc[grid](
            x_nhwc, out, self.bias,
            N, C, H, W,
            BLOCK_C=BLOCK_C,
            BLOCK_W=BLOCK_W,
            num_warps=8,
            num_stages=2,
        )
        return out