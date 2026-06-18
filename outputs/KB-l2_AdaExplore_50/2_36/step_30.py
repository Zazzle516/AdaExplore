import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_conv_transpose_min_sum_gelu_kernel(
    x_ptr,           # [N, IC, H_in, W_in]
    w_ptr,           # [IC, OC, KH, KW]
    conv_bias_ptr,   # [OC]
    bias_ptr,        # scalar
    out_ptr,         # [N, 1, 1, W_out]
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE: tl.constexpr, PAD: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    # Each program: one (n, w_block) over W_out
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W_out

    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    INF = float('inf')

    # Accumulator: sum over h_out of min over oc.
    # We sum per (w_out) position.
    sum_acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    # For each h_out, accumulate conv_transpose output across (oc), then min-reduce, then add to sum.
    for h_out in range(0, H_out):
        # Initialize acc per oc per w: starts at conv bias
        # conv_bias for each oc broadcast to all w
        cb = tl.load(conv_bias_ptr + oc_offs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
        acc = cb[:, None] + tl.zeros([BLOCK_OC, BLOCK_W], dtype=tl.float32)

        # ConvTranspose2d: out[n, oc, h_out, w_out] = sum_{ic, kh, kw}
        #   x[n, ic, h_in, w_in] * w[ic, oc, kh, kw]
        # where h_in*stride - pad + kh = h_out  -> kh = h_out + pad - h_in*stride
        # We iterate over kh, kw and compute h_in, w_in.
        for kh in range(0, KH):
            # h_in_num = h_out + PAD - kh; must be divisible by STRIDE and in [0, H_in)
            h_in_num = h_out + PAD - kh
            h_in = h_in_num // STRIDE
            h_valid = (h_in_num >= 0) & (h_in_num % STRIDE == 0) & (h_in < H_in) & (h_in >= 0)

            for kw in range(0, KW):
                w_in_num = w_offs + PAD - kw  # [BLOCK_W]
                w_in = w_in_num // STRIDE
                w_valid = (w_in_num >= 0) & ((w_in_num % STRIDE) == 0) & (w_in < W_in) & (w_in >= 0) & w_mask

                if h_valid:
                    # Load x[n, :, h_in, w_in] for all ic, all w in block
                    # x layout: [N, IC, H_in, W_in], contiguous
                    # We loop over ic, but better: load slice and gemm. For simplicity loop over ic in tiles.
                    # weight w[ic, oc, kh, kw] : [IC, OC, KH, KW]
                    # We need contribution: sum_ic x[n,ic,h_in,w_in] * w[ic, oc, kh, kw]
                    # shape: x_slice [IC, BLOCK_W], w_slice [IC, BLOCK_OC] -> acc [BLOCK_OC, BLOCK_W] += w_slice.T @ x_slice
                    for ic in range(0, IC):
                        x_ptrs = x_ptr + n_offset_dummy(pid_n) + ic * H_in * W_in + h_in * W_in + w_in
                        xv = tl.load(x_ptrs, mask=w_valid, other=0.0)  # [BLOCK_W]
                        w_ptrs = w_ptr + ic * OC * KH * KW + oc_offs * KH * KW + kh * KW + kw
                        wv = tl.load(w_ptrs, mask=oc_mask, other=0.0)  # [BLOCK_OC]
                        acc += wv[:, None] * xv[None, :]

        # Mask invalid oc to +inf so they don't affect min
        acc = tl.where(oc_mask[:, None], acc, INF)
        # Min over OC
        mn = tl.min(acc, axis=0)  # [BLOCK_W]
        sum_acc += mn

    # GELU (exact)
    g = 0.5 * sum_acc * (1.0 + tl.erf(sum_acc * 0.70710678118654752440))
    b = tl.load(bias_ptr)
    res = g + b

    out_offs = pid_n * W_out + w_offs
    tl.store(out_ptr + out_offs, res, mask=w_mask)


@triton.jit
def n_offset_dummy(n):
    # placeholder, never used because we inline below
    return n


# The above approach is too complex; use a simpler fused min+sum+gelu after using torch's conv_transpose.

@triton.jit
def min_sum_gelu_bias_kernel(
    x_ptr,         # input: [N, C, H, W]
    out_ptr,       # output: [N, 1, 1, W]
    bias_ptr,      # bias scalar
    N, C, H, W,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_w = tl.program_id(1)

    w_offs = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    w_mask = w_offs < W

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    INF = float('inf')

    sum_acc = tl.zeros([BLOCK_W], dtype=tl.float32)

    base_n = pid_n * C * H * W

    for h in range(0, H):
        # Load x[n, :, h, w_offs]: shape [BLOCK_C, BLOCK_W]
        ptrs = x_ptr + base_n + c_offs[:, None] * H * W + h * W + w_offs[None, :]
        mask = c_mask[:, None] & w_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=INF)
        mn = tl.min(vals, axis=0)  # [BLOCK_W]
        sum_acc += mn

    g = 0.5 * sum_acc * (1.0 + tl.erf(sum_acc * 0.70710678118654752440))
    b = tl.load(bias_ptr)
    res = g + b

    out_offs = pid_n * W + w_offs
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
        BLOCK_W = 64
        if W < BLOCK_W:
            BLOCK_W = triton.next_power_of_2(W)

        grid = (N, (W + BLOCK_W - 1) // BLOCK_W)
        min_sum_gelu_bias_kernel[grid](
            x, out, self.bias,
            N, C, H, W,
            BLOCK_W=BLOCK_W,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out