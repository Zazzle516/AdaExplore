import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy:
#  - Use cuDNN F.conv3d for the convolution (highly optimized).
#  - Write a single fused Triton kernel that does:
#      divide by divisor, 2x2x2 maxpool, sum over all pool windows (numerator
#      of global average), divide by num_windows, add bias[oc]
#    producing one scalar per (N, OC).
#  - Final sum over OC done in PyTorch (tiny).
#
# Conv output shape: [N, OC, D_out=14, H_out=62, W_out=62]
# After 2x2x2 maxpool: [N, OC, 7, 31, 31]  (W_out=62 -> 31 windows; trailing element ignored as in nn.MaxPool3d default)


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=2),
    ],
    key=['D_out', 'H_out', 'W_out'],
)
@triton.jit
def fused_post_conv_kernel(
    conv_ptr,       # [N, OC, D_out, H_out, W_out]
    bias_ptr,       # [OC]
    out_ptr,        # [N, OC]
    N, OC,
    D_out: tl.constexpr, H_out: tl.constexpr, W_out: tl.constexpr,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    inv_div: tl.constexpr,
    inv_pool: tl.constexpr,
    BLOCK_PW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    # base pointer to this (n, oc) slice of conv output
    slice_size = D_out * H_out * W_out
    base = pid_n * OC * slice_size + pid_oc * slice_size

    bias_val = tl.load(bias_ptr + pid_oc)

    acc = 0.0

    pw_offs = tl.arange(0, BLOCK_PW)  # pool-window indices along W
    pw_mask = pw_offs < PW

    # Each (pd, ph) tile processes all PW pool windows along W in parallel
    for pd in range(0, PD):
        d0 = pd * 2
        d1 = d0 + 1
        for ph in range(0, PH):
            h0 = ph * 2
            h1 = h0 + 1
            # W indices: each pool window pw covers W positions [2*pw, 2*pw+1]
            w0 = pw_offs * 2
            w1 = w0 + 1

            # Load 8 neighbors per pool window, vectorized over BLOCK_PW
            # Index = base + d * (H_out*W_out) + h * W_out + w
            HW = H_out * W_out
            p000 = tl.load(conv_ptr + base + d0 * HW + h0 * W_out + w0, mask=pw_mask, other=-float('inf'))
            p001 = tl.load(conv_ptr + base + d0 * HW + h0 * W_out + w1, mask=pw_mask, other=-float('inf'))
            p010 = tl.load(conv_ptr + base + d0 * HW + h1 * W_out + w0, mask=pw_mask, other=-float('inf'))
            p011 = tl.load(conv_ptr + base + d0 * HW + h1 * W_out + w1, mask=pw_mask, other=-float('inf'))
            p100 = tl.load(conv_ptr + base + d1 * HW + h0 * W_out + w0, mask=pw_mask, other=-float('inf'))
            p101 = tl.load(conv_ptr + base + d1 * HW + h0 * W_out + w1, mask=pw_mask, other=-float('inf'))
            p110 = tl.load(conv_ptr + base + d1 * HW + h1 * W_out + w0, mask=pw_mask, other=-float('inf'))
            p111 = tl.load(conv_ptr + base + d1 * HW + h1 * W_out + w1, mask=pw_mask, other=-float('inf'))

            # Apply divisor
            p000 = p000 * inv_div
            p001 = p001 * inv_div
            p010 = p010 * inv_div
            p011 = p011 * inv_div
            p100 = p100 * inv_div
            p101 = p101 * inv_div
            p110 = p110 * inv_div
            p111 = p111 * inv_div

            m = tl.maximum(p000, p001)
            m = tl.maximum(m, p010)
            m = tl.maximum(m, p011)
            m = tl.maximum(m, p100)
            m = tl.maximum(m, p101)
            m = tl.maximum(m, p110)
            m = tl.maximum(m, p111)

            # Mask out invalid lanes before sum
            m = tl.where(pw_mask, m, 0.0)
            acc += tl.sum(m, axis=0)

    # Global avg pool: divide by num windows, then add bias
    result = acc * inv_pool + bias_val
    tl.store(out_ptr + pid_n * OC + pid_oc, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        # Pre-compute flat bias view shape info
        self._oc = out_channels

    def forward(self, x):
        N = x.shape[0]
        OC = self.out_channels

        # cuDNN conv3d
        y = F.conv3d(x, self.conv.weight, self.conv.bias)
        # y: [N, OC, D_out, H_out, W_out]
        _, _, D_out, H_out, W_out = y.shape

        PSD, PSH, PSW = self.pool_size
        PD = D_out // PSD
        PH = H_out // PSH
        PW = W_out // PSW

        bias_flat = self.bias.view(OC)

        out = torch.empty((N, OC), device=x.device, dtype=y.dtype)

        inv_div = 1.0 / float(self.divisor)
        inv_pool = 1.0 / float(PD * PH * PW)

        # BLOCK_PW must be a power of 2 >= PW (31 -> 32)
        BLOCK_PW = 32
        while BLOCK_PW < PW:
            BLOCK_PW *= 2

        grid = (N, OC)
        fused_post_conv_kernel[grid](
            y, bias_flat, out,
            N, OC,
            D_out, H_out, W_out,
            PD, PH, PW,
            inv_div, inv_pool,
            BLOCK_PW,
        )

        # Sum over OC (dim=1 of original [N, OC, 1, 1, 1] -> [N, 1, 1, 1])
        if self.sum_dim == 1:
            result = out.sum(dim=1).view(N, 1, 1, 1)
        else:
            full = out.view(N, OC, 1, 1, 1)
            result = full.sum(dim=self.sum_dim)
        return result