import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Notes on shapes:
# Input: (B, C_in=64, H=256, W=256)
# ConvTranspose2d(kernel=3, stride=1, padding=1) -> (B, C_out=64, 256, 256)
# MaxPool2d(2,2) -> (B, 64, 128, 128)
# Hardtanh(-1, 1), mean over (H,W) -> (B, 64, 1, 1), tanh
#
# Strategy:
#  - Do the ConvTranspose2d using PyTorch (cuDNN is highly optimized).
#  - Fuse maxpool(2,2) + hardtanh + spatial mean + tanh into ONE triton kernel.
#    For each (n, c), reduce over the 128x128 pooled output by computing
#    max over each 2x2 window of conv output, clamping to [-1,1], summing,
#    then dividing by pooled_H*pooled_W and applying tanh.


@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr,            # conv output (B, C, H, W) contiguous
    out_ptr,          # (B, C) output (will be reshaped to B,C,1,1)
    B, C, H, W,
    pooled_H, pooled_W,
    inv_count,        # 1.0 / (pooled_H * pooled_W)
    hmin, hmax,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, c)
    n = pid // C
    c = pid % C

    # base pointer into x for this (n, c)
    base = n * C * H * W + c * H * W

    total = pooled_H * pooled_W  # number of pooled positions
    acc = 0.0

    offs = tl.arange(0, BLOCK)
    num_blocks = (total + BLOCK - 1) // BLOCK

    for b in range(0, num_blocks):
        idx = b * BLOCK + offs
        mask = idx < total
        ph = idx // pooled_W
        pw = idx - ph * pooled_W
        # 2x2 window: input rows ph*2, ph*2+1, cols pw*2, pw*2+1
        h0 = ph * 2
        w0 = pw * 2

        p00 = base + h0 * W + w0
        p01 = base + h0 * W + (w0 + 1)
        p10 = base + (h0 + 1) * W + w0
        p11 = base + (h0 + 1) * W + (w0 + 1)

        v00 = tl.load(x_ptr + p00, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + p01, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + p10, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + p11, mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, hmin), hmax)
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    mean_val = acc * inv_count
    # tanh
    # tl.tanh may not be available; use formula via exp
    e1 = tl.exp(mean_val)
    e2 = tl.exp(-mean_val)
    out = (e1 - e2) / (e1 + e2)

    tl.store(out_ptr + pid, out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                 stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)
        # also keep these as modules for compatibility
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        B, C, H, W = x.shape

        # We assume maxpool kernel == stride == 2 (per the configured params).
        # Fallback to torch implementation if not.
        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and (H % 2 == 0) and (W % 2 == 0):
            pooled_H = H // 2
            pooled_W = W // 2
            out = torch.empty((B, C), device=x.device, dtype=x.dtype)
            grid = (B * C,)
            inv_count = 1.0 / (pooled_H * pooled_W)
            BLOCK = 1024
            fused_pool_htanh_mean_tanh_kernel[grid](
                x, out,
                B, C, H, W,
                pooled_H, pooled_W,
                inv_count,
                self.hardtanh_min, self.hardtanh_max,
                BLOCK=BLOCK,
                num_warps=4,
            )
            return out.view(B, C, 1, 1)
        else:
            x = self.maxpool(x)
            x = self.hardtanh(x)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x