import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# With stride=1, padding=1, kernel=3, ConvTranspose2d produces output of same spatial size as input.
# Then maxpool 2x2 stride 2 -> H/2, W/2.
# Then hardtanh, mean, tanh.
# Output shape: (N, C_out, 1, 1)
# We fuse maxpool+hardtanh+mean+tanh into a single kernel that reads conv output.

@triton.jit
def fused_pool_htanh_mean_tanh_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    HTMIN: tl.constexpr,
    HTMAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W

    total = H_out * W_out
    inv = 1.0 / total

    acc = tl.zeros([], dtype=tl.float32)

    # iterate over pooled positions in blocks
    for start in range(0, total, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < total
        ho = offs // W_out
        wo = offs % W_out
        h0 = ho * 2
        w0 = wo * 2

        # load 4 elements of 2x2 window
        i00 = base + h0 * W + w0
        i01 = base + h0 * W + (w0 + 1)
        i10 = base + (h0 + 1) * W + w0
        i11 = base + (h0 + 1) * W + (w0 + 1)

        v00 = tl.load(x_ptr + i00, mask=mask, other=-1e30)
        v01 = tl.load(x_ptr + i01, mask=mask, other=-1e30)
        v10 = tl.load(x_ptr + i10, mask=mask, other=-1e30)
        v11 = tl.load(x_ptr + i11, mask=mask, other=-1e30)

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, HTMIN), HTMAX)
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    mean_val = acc * inv
    # tanh
    e2x = tl.exp(2.0 * mean_val)
    out_val = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + pid, out_val)


def fused_pool_htanh_mean_tanh(x: torch.Tensor, hardtanh_min: float, hardtanh_max: float):
    N, C, H, W = x.shape
    H_out = H // 2
    W_out = W // 2
    x = x.contiguous()
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    BLOCK = 1024
    fused_pool_htanh_mean_tanh_kernel[grid](
        x, out,
        N, C, H, W,
        H_out, W_out,
        float(hardtanh_min), float(hardtanh_max),
        BLOCK=BLOCK,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = hardtanh_min
        self.hardtanh_max = hardtanh_max

    def forward(self, x):
        x = self.conv_transpose(x)
        # Only use fused kernel if pool params match assumptions (2x2 stride 2)
        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and x.shape[2] % 2 == 0 and x.shape[3] % 2 == 0:
            return fused_pool_htanh_mean_tanh(x, self.hardtanh_min, self.hardtanh_max)
        else:
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x