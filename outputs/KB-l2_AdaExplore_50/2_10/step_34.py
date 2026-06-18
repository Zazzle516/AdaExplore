import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Custom ConvTranspose2d for stride=1, padding=1, kernel=3
# This is equivalent to conv2d with flipped weights and padding=1.
# We implement it as an im2col-style GEMM fused with the maxpool+hardtanh+mean+tanh tail.
# Strategy: produce conv output and immediately reduce via pool+htanh+mean+tanh
# without materializing the 128x64x256x256 intermediate.

@triton.jit
def fused_conv_pool_htanh_mean_tanh_kernel(
    x_ptr,          # input [N, IC, H, W]
    w_ptr,          # weight [IC, OC, 3, 3] (ConvTranspose2d layout)
    b_ptr,          # bias [OC]
    out_ptr,        # output [N, OC, 1, 1]
    N, IC, OC, H, W,
    H_OUT2: tl.constexpr,  # H // 2
    W_OUT2: tl.constexpr,  # W // 2
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    INV_COUNT: tl.constexpr,
    BLOCK_HW: tl.constexpr,  # number of (oh_pool, ow_pool) per program tile
):
    # Each program: one (n, oc), reduces over all pooled spatial positions.
    pid = tl.program_id(0)
    n = pid // OC
    oc = pid % OC

    # Load bias
    bias_val = tl.load(b_ptr + oc)

    # Constants
    TOTAL: tl.constexpr = H_OUT2 * W_OUT2

    offs = tl.arange(0, BLOCK_HW)
    acc_sum = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Pre-extract weight: weight layout is [IC, OC, 3, 3]
    # For "conv-transpose with stride=1,padding=1,k=3" -> equivalent forward correlation with kernel flipped.
    # Output(n, oc, y, x) = sum_ic sum_ky,kx Input(n, ic, y + ky - 1, x + kx - 1) * W_flipped(ic, oc, ky, kx)
    # where W_flipped(ic, oc, ky, kx) = W(ic, oc, 2-ky, 2-kx)

    # We iterate over output pooled positions in chunks.
    for start in range(0, TOTAL, BLOCK_HW):
        idx = start + offs
        mask_idx = idx < TOTAL

        # pooled output positions
        poh = idx // W_OUT2
        pow_ = idx % W_OUT2

        # corresponding 2x2 output positions in conv output
        oy0 = poh * 2
        ox0 = pow_ * 2
        # Four output points: (oy0, ox0), (oy0, ox0+1), (oy0+1, ox0), (oy0+1, ox0+1)

        # Initialize 4 accumulators with bias
        c00 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val
        c01 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val
        c10 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val
        c11 = tl.zeros([BLOCK_HW], dtype=tl.float32) + bias_val

        # Loop over input channels and kernel
        for ic in range(0, IC):
            x_base = n * (IC * H * W) + ic * (H * W)
            w_base = ic * (OC * 9) + oc * 9  # [IC, OC, 3, 3]

            # Load 9 weights (flipped: kernel idx (ky,kx) uses weight at (2-ky, 2-kx))
            # We'll instead iterate kernel positions and use flipped weight index.
            for ky in tl.static_range(0, 3):
                for kx in tl.static_range(0, 3):
                    w_val = tl.load(w_ptr + w_base + (2 - ky) * 3 + (2 - kx))

                    # For each of 4 outputs, compute input position
                    # Input(n, ic, oy + ky - 1, ox + kx - 1)
                    for dy in tl.static_range(0, 2):
                        for dx in tl.static_range(0, 2):
                            oy = oy0 + dy
                            ox = ox0 + dx
                            iy = oy + ky - 1
                            ix = ox + kx - 1
                            in_bounds = (iy >= 0) & (iy < H) & (ix >= 0) & (ix < W) & mask_idx
                            ptr = x_base + iy * W + ix
                            v = tl.load(x_ptr + ptr, mask=in_bounds, other=0.0)
                            contrib = v * w_val
                            if dy == 0 and dx == 0:
                                c00 += contrib
                            if dy == 0 and dx == 1:
                                c01 += contrib
                            if dy == 1 and dx == 0:
                                c10 += contrib
                            if dy == 1 and dx == 1:
                                c11 += contrib

        # Max pool 2x2
        m = tl.maximum(tl.maximum(c00, c01), tl.maximum(c10, c11))
        # Hardtanh
        m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
        m = tl.where(mask_idx, m, 0.0)
        acc_sum += m

    s = tl.sum(acc_sum, axis=0)
    mean = s * INV_COUNT
    e2 = tl.exp(2.0 * mean)
    out_val = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + pid, out_val)


# Tail-only kernel (used if we still rely on torch's conv_transpose)
@triton.jit
def fused_pool_htanh_mean_kernel(
    x_ptr,
    out_ptr,
    NC,
    H_IN: tl.constexpr,
    W_IN: tl.constexpr,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * H_IN * W_IN
    TOTAL: tl.constexpr = H_OUT * W_OUT
    INV_COUNT: tl.constexpr = 1.0 / (H_OUT * W_OUT)

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for start in range(0, TOTAL, BLOCK):
        idx = start + offs
        mask = idx < TOTAL
        oh = idx // W_OUT
        ow = idx % W_OUT

        ih0 = oh * 2
        iw0 = ow * 2

        row0 = base + ih0 * W_IN + iw0
        row1 = row0 + W_IN

        v00 = tl.load(x_ptr + row0, mask=mask, other=0.0)
        v01 = tl.load(x_ptr + row0 + 1, mask=mask, other=0.0)
        v10 = tl.load(x_ptr + row1, mask=mask, other=0.0)
        v11 = tl.load(x_ptr + row1 + 1, mask=mask, other=0.0)

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
        acc += m

    s = tl.sum(acc, axis=0)
    mean = s * INV_COUNT
    e2 = tl.exp(2.0 * mean)
    out_val = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + pid, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, H, W = x.shape

        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and H % 2 == 0 and W % 2 == 0:
            H_out = H // 2
            W_out = W // 2
            out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
            BLOCK = 1024
            grid = (N * C,)
            fused_pool_htanh_mean_kernel[grid](
                x, out,
                N * C,
                H, W,
                H_out, W_out,
                self.hardtanh_min, self.hardtanh_max,
                BLOCK=BLOCK,
                num_warps=4,
                num_stages=2,
            )
            return out
        else:
            x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            x = F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x