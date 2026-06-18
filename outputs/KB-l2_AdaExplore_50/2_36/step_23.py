import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def partial_min_sum_nhwc_kernel(
    x_ptr, partial_ptr,
    N, C, H, W, NUM_CHUNKS,
    BLOCK_C: tl.constexpr,
    H_CHUNK: tl.constexpr,
):
    # NHWC layout. Grid: (N * W, num_h_chunks)
    # Each program computes sum over a chunk of H of min-over-C at (n,h,w)
    pid_nw = tl.program_id(0)
    pid_h = tl.program_id(1)

    n = pid_nw // W
    w = pid_nw % W

    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    h_start = pid_h * H_CHUNK
    s = 0.0
    for hi in range(0, H_CHUNK):
        h = h_start + hi
        valid = h < H
        # NHWC offset: n*H*W*C + h*W*C + w*C + c
        ptrs = x_ptr + n * H * W * C + h * W * C + w * C + offs_c
        x = tl.load(ptrs, mask=c_mask & valid, other=float('inf'))
        m = tl.min(x, axis=0)
        s += tl.where(valid, m, 0.0)

    # partial_ptr layout: (N, W, NUM_CHUNKS)
    tl.store(partial_ptr + n * W * NUM_CHUNKS + w * NUM_CHUNKS + pid_h, s)


@triton.jit
def final_sum_gelu_bias_kernel(
    partial_ptr, out_ptr, bias_ptr,
    N, W, NUM_CHUNKS,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // W
    w = pid % W

    offs = tl.arange(0, BLOCK)
    mask = offs < NUM_CHUNKS
    base = n * W * NUM_CHUNKS + w * NUM_CHUNKS
    v = tl.load(partial_ptr + base + offs, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)

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
        # Convert conv weights to channels_last for NHWC fast path
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        # x is in channels_last (NHWC) memory format
        N, C, H, W = x.shape

        H_CHUNK = 32
        num_chunks = (H + H_CHUNK - 1) // H_CHUNK
        partial = torch.empty((N, W, num_chunks), device=x.device, dtype=x.dtype)

        BLOCK_C = _next_pow2(C)

        partial_min_sum_nhwc_kernel[(N * W, num_chunks)](
            x, partial,
            N, C, H, W, num_chunks,
            BLOCK_C=BLOCK_C,
            H_CHUNK=H_CHUNK,
            num_warps=4, num_stages=2,
        )

        out = torch.empty((N, 1, 1, W), device=x.device, dtype=x.dtype)
        BLOCK = _next_pow2(num_chunks)
        final_sum_gelu_bias_kernel[(N * W,)](
            partial, out, self.bias,
            N, W, num_chunks,
            BLOCK=BLOCK,
            num_warps=1, num_stages=2,
        )

        return out