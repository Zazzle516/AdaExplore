import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# With kernel_size=3, stride=1, padding=1, ConvTranspose2d gives same H,W.
# Then maxpool 2x2 stride 2 halves H,W. Then mean over (H/2)*(W/2).
# Fuse: conv_transpose -> maxpool(2x2) -> hardtanh -> mean -> tanh
# Strategy: do conv_transpose with torch (fast cuDNN), then fuse the rest in one kernel.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
    ],
    key=['H', 'W'],
)
@triton.jit
def _pool_htanh_mean_tanh_kernel(
    x_ptr,        # [N, C, H, W]
    out_ptr,      # [N, C, 1, 1]
    N, C, H, W,
    pH, pW,       # pooled H, W (H//2, W//2)
    hmin, hmax,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W
    total = pH * pW  # number of pooled output elements
    inv_total = 1.0 / total.to(tl.float32)

    acc = 0.0
    for off in range(0, total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total
        ph = idx // pW
        pw = idx % pW
        # 2x2 window starting at (2*ph, 2*pw)
        h0 = 2 * ph
        w0 = 2 * pw

        v00 = tl.load(x_ptr + base + (h0 + 0) * W + (w0 + 0), mask=mask, other=-1e30)
        v01 = tl.load(x_ptr + base + (h0 + 0) * W + (w0 + 1), mask=mask, other=-1e30)
        v10 = tl.load(x_ptr + base + (h0 + 1) * W + (w0 + 0), mask=mask, other=-1e30)
        v11 = tl.load(x_ptr + base + (h0 + 1) * W + (w0 + 1), mask=mask, other=-1e30)

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, hmin), hmax)
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)

    mean_val = acc * inv_total
    # tanh
    out_val = (tl.exp(2.0 * mean_val) - 1.0) / (tl.exp(2.0 * mean_val) + 1.0)
    tl.store(out_ptr + n * C + c, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding
        )
        self.maxpool_kernel_size = maxpool_kernel_size
        self.maxpool_stride = maxpool_stride
        self.hardtanh_min = float(hardtanh_min)
        self.hardtanh_max = float(hardtanh_max)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x.contiguous()
        N, C, H, W = x.shape

        # Fall back to torch if shapes don't match the fused kernel's assumptions
        if (self.maxpool_kernel_size == 2 and self.maxpool_stride == 2
                and H % 2 == 0 and W % 2 == 0):
            pH = H // 2
            pW = W // 2
            out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
            grid = (N * C,)
            _pool_htanh_mean_tanh_kernel[grid](
                x, out,
                N, C, H, W, pH, pW,
                self.hardtanh_min, self.hardtanh_max,
            )
            return out
        else:
            x = F.max_pool2d(x, self.maxpool_kernel_size, self.maxpool_stride)
            x = F.hardtanh(x, self.hardtanh_min, self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x