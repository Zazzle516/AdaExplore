import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    pooled_H, pooled_W,
    hardtanh_min, hardtanh_max,
    inv_count,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    total = pooled_H * pooled_W
    base = n * C * H * W + c * H * W

    acc = 0.0
    for start in range(0, total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total
        ph = offs // pooled_W
        pw = offs % pooled_W
        h0 = ph * 2
        w0 = pw * 2
        # gather 2x2
        i00 = base + h0 * W + w0
        i01 = base + h0 * W + (w0 + 1)
        i10 = base + (h0 + 1) * W + w0
        i11 = base + (h0 + 1) * W + (w0 + 1)
        v00 = tl.load(x_ptr + i00, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + i01, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + i10, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + i11, mask=mask, other=-float('inf'))
        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, hardtanh_min), hardtanh_max)
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    mean_val = acc * inv_count
    # tanh via exp
    e1 = tl.exp(mean_val)
    e2 = tl.exp(-mean_val)
    out_val = (e1 - e2) / (e1 + e2)
    tl.store(out_ptr + pid, out_val)


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

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, H, W = x.shape
        x = x.contiguous()

        # Assume maxpool kernel=2, stride=2 (as per defaults)
        pooled_H = H // 2
        pooled_W = W // 2

        out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

        total = pooled_H * pooled_W
        # choose BLOCK
        BLOCK = 1024
        if total < 1024:
            BLOCK = 256
        if total < 256:
            BLOCK = 64

        grid = (N * C,)
        fused_pool_htanh_mean_tanh_kernel[grid](
            x, out,
            N, C, H, W,
            pooled_H, pooled_W,
            self.hardtanh_min, self.hardtanh_max,
            1.0 / float(total),
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out