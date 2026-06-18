import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_min_sum_gelu_bias_kernel(
    x_ptr, out_ptr, bias_ptr,
    N, C, H, W,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, w); loops over H, reducing min over C inline
    pid = tl.program_id(0)
    n = pid // W
    w = pid % W

    HW = H * W
    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    base_nw = n * C * HW + w

    s = 0.0
    for h in range(0, H):
        ptrs = x_ptr + base_nw + offs_c * HW + h * W
        x = tl.load(ptrs, mask=c_mask, other=float('inf'))
        m = tl.min(x, axis=0)
        s += m

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

        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)
        BLOCK_C = _next_pow2(C)
        grid = (N * W,)
        fused_min_sum_gelu_bias_kernel[grid](
            x, out, self.bias,
            N, C, H, W,
            BLOCK_C=BLOCK_C,
            num_warps=4, num_stages=2,
        )
        return out