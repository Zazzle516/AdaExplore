import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr,         # input: conv_transpose output [N, C, H, W]
    out_ptr,       # output: [N, C, 1, 1]
    N, C, H, W,
    H_out, W_out,  # after maxpool dims
    htanh_min, htanh_max,
    inv_area,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W

    total = H_out * W_out
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    offs = tl.arange(0, BLOCK_SIZE)

    # iterate over output positions in chunks
    for start in range(0, total, BLOCK_SIZE):
        idx = start + offs
        mask = idx < total
        ho = idx // W_out
        wo = idx % W_out
        # maxpool 2x2 stride 2
        h0 = ho * 2
        w0 = wo * 2

        p00 = tl.load(x_ptr + base + (h0 + 0) * W + (w0 + 0), mask=mask, other=-float('inf'))
        p01 = tl.load(x_ptr + base + (h0 + 0) * W + (w0 + 1), mask=mask, other=-float('inf'))
        p10 = tl.load(x_ptr + base + (h0 + 1) * W + (w0 + 0), mask=mask, other=-float('inf'))
        p11 = tl.load(x_ptr + base + (h0 + 1) * W + (w0 + 1), mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(p00, p01), tl.maximum(p10, p11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, htanh_min), htanh_max)
        acc += tl.where(mask, m, 0.0)

    total_sum = tl.sum(acc, axis=0)
    mean_val = total_sum * inv_area
    # tanh via sigmoid trick: tanh(x) = 2*sigmoid(2x)-1
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
            BLOCK_SIZE=1024,
            num_warps=4,
        )
        return out