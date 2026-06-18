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
    # x: (N, H, W, C) channels-last
    # For each (n, w): sum_h min_c x[n,h,w,c], then GELU + bias
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    offs_w = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = offs_w < W

    # base for (n, h=0, w=0, c=0)
    n_base = pid_n * H * W * C
    acc = tl.zeros((BLOCK_W,), dtype=tl.float32)

    for h in range(0, H):
        # x[n, h, w, c] = n_base + h*W*C + w*C + c
        offs = (h * W * C) + (offs_w[:, None] * C) + offs_c[None, :]
        mask = w_mask[:, None] & c_mask[None, :]
        vals = tl.load(x_ptr + n_base + offs, mask=mask, other=float('inf'))
        m = tl.min(vals, axis=1)  # [BLOCK_W]
        acc = acc + m

    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * acc * (1.0 + tl.erf(acc * inv_sqrt2))

    b = tl.load(bias_ptr)
    out = g + b
    tl.store(out_ptr + pid_n * W + offs_w, out, mask=w_mask)


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
        # Convert to channels-last so C is contiguous for min reduction
        x_cl = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)

        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)
        BLOCK_C = _next_pow2(C)
        BLOCK_W = 128 if W >= 128 else _next_pow2(W)
        grid = (N, triton.cdiv(W, BLOCK_W))
        fused_min_sum_gelu_bias_kernel[grid](
            x_cl, out, self.bias,
            N, C, H, W,
            BLOCK_C=BLOCK_C, BLOCK_W=BLOCK_W,
            num_warps=8, num_stages=2,
        )

        return out