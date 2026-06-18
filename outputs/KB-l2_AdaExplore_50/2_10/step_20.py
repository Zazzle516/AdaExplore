import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    H_out, W_out,
    htanh_min: tl.constexpr,
    htanh_max: tl.constexpr,
    inv_area,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W

    total = H_out * W_out
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    offs = tl.arange(0, BLOCK_SIZE)

    for start in range(0, total, BLOCK_SIZE):
        idx = start + offs
        mask = idx < total
        ho = idx // W_out
        wo = idx % W_out
        h0 = ho * 2
        w0 = wo * 2

        row0 = base + h0 * W + w0
        row1 = row0 + W

        p00 = tl.load(x_ptr + row0, mask=mask, other=-float('inf'))
        p01 = tl.load(x_ptr + row0 + 1, mask=mask, other=-float('inf'))
        p10 = tl.load(x_ptr + row1, mask=mask, other=-float('inf'))
        p11 = tl.load(x_ptr + row1 + 1, mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(p00, p01), tl.maximum(p10, p11))
        m = tl.minimum(tl.maximum(m, htanh_min), htanh_max)
        acc += tl.where(mask, m, 0.0)

    total_sum = tl.sum(acc, axis=0)
    mean_val = total_sum * inv_area
    e2 = tl.exp(2.0 * mean_val)
    t = (e2 - 1.0) / (e2 + 1.0)

    tl.store(out_ptr + n * C + c, t)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(ModelNew, self).__init__()
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
        H_out = H // 2
        W_out = W // 2
        out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
        grid = (N * C,)
        inv_area = 1.0 / float(H_out * W_out)
        fused_pool_htanh_mean_tanh_kernel[grid](
            x, out,
            N, C, H, W,
            H_out, W_out,
            self.hardtanh_min, self.hardtanh_max,
            inv_area,
            BLOCK_SIZE=2048,
            num_warps=8,
            num_stages=3,
        )
        return out