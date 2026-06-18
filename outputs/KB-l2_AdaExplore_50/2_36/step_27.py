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
    bias_ptr,      # bias scalar (1,1,1)
    N, C, H, W,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, w)
    pid = tl.program_id(0)
    n = pid // W
    w = pid % W

    # accumulate sum over H of min over C
    h_acc = tl.zeros([], dtype=tl.float32)
    
    INF = float('inf')
    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    sum_val = 0.0
    for h in range(0, H):
        # load x[n, :, h, w]
        ptrs = x_ptr + n * C * H * W + c_offs * H * W + h * W + w
        vals = tl.load(ptrs, mask=c_mask, other=INF)
        mn = tl.min(vals, axis=0)
        sum_val = sum_val + mn

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

        grid = (N * W,)
        min_sum_gelu_bias_kernel[grid](
            x, out, self.bias,
            N, C, H, W,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out