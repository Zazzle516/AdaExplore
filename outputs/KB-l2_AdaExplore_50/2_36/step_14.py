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
    BLOCK_W: tl.constexpr,
):
    # one program per (n, w_block); loop over H, take min over C, accumulate sum
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_start = pid_w * BLOCK_W
    offs_w = w_start + tl.arange(0, BLOCK_W)
    mask_w = offs_w < W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    CHW = C * H * W
    HW = H * W

    s = tl.zeros([BLOCK_W], dtype=tl.float32)
    base_n = pid_n * CHW
    # 2D tile pointers: (BLOCK_C, BLOCK_W)
    c_off = offs_c[:, None] * HW
    w_off = offs_w[None, :]
    mask_2d = mask_c[:, None] & mask_w[None, :]

    for h in range(0, H):
        ptrs = x_ptr + base_n + c_off + h * W + w_off
        x = tl.load(ptrs, mask=mask_2d, other=float('inf'))
        m = tl.min(x, axis=0)  # (BLOCK_W,)
        s += m

    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * s * (1.0 + tl.erf(s * inv_sqrt2))
    b = tl.load(bias_ptr)
    out_off = pid_n * W + offs_w
    tl.store(out_ptr + out_off, g + b, mask=mask_w)


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
        BLOCK_W = 8
        grid = (N, triton.cdiv(W, BLOCK_W))
        fused_min_sum_gelu_bias_kernel[grid](
            x, out, self.bias, N, C, H, W,
            BLOCK_C=BLOCK_C, BLOCK_W=BLOCK_W, num_warps=4, num_stages=2,
        )
        return out