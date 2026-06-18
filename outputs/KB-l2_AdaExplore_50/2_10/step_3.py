import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# With kernel_size=3, stride=1, padding=1, ConvTranspose2d output has same H,W as input.
# Then MaxPool2d(2,2) halves H,W.
# Then mean over (H/2, W/2) and tanh.
# We fuse: maxpool + hardtanh + mean over spatial into one kernel.

@triton.jit
def fused_pool_htanh_mean_kernel(
    x_ptr,        # input after conv_transpose: [N, C, H, W]
    out_ptr,      # output: [N, C, 1, 1]
    N, C, H, W,
    H_out, W_out, # H//2, W//2
    inv_count,    # 1.0 / (H_out * W_out)
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, c)
    n = pid // C
    c = pid % C

    # Pointer to start of (n, c) plane
    base = n * C * H * W + c * H * W
    total = H_out * W_out

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    # iterate over output positions in chunks of BLOCK
    for start in range(0, total, BLOCK):
        idx = start + offs
        mask = idx < total
        oh = idx // W_out
        ow = idx % W_out

        # Each output corresponds to 2x2 input window starting at (oh*2, ow*2)
        ih0 = oh * 2
        iw0 = ow * 2

        p00 = base + ih0 * W + iw0
        p01 = base + ih0 * W + (iw0 + 1)
        p10 = base + (ih0 + 1) * W + iw0
        p11 = base + (ih0 + 1) * W + (iw0 + 1)

        v00 = tl.load(x_ptr + p00, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + p01, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + p10, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + p11, mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        # hardtanh
        m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
        m = tl.where(mask, m, 0.0)
        acc += m

    s = tl.sum(acc, axis=0)
    mean = s * inv_count
    # tanh
    out_val = (tl.exp(2.0 * mean) - 1.0) / (tl.exp(2.0 * mean) + 1.0)
    tl.store(out_ptr + pid, out_val)


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
        x = x.contiguous()
        N, C, H, W = x.shape

        # Only fast-path the typical config: maxpool 2x2 stride 2
        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and H % 2 == 0 and W % 2 == 0:
            H_out = H // 2
            W_out = W // 2
            out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
            inv_count = 1.0 / (H_out * W_out)
            BLOCK = 1024
            grid = (N * C,)
            fused_pool_htanh_mean_kernel[grid](
                x, out,
                N, C, H, W,
                H_out, W_out,
                inv_count,
                self.hardtanh_min, self.hardtanh_max,
                BLOCK=BLOCK,
                num_warps=4,
            )
            return out
        else:
            x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            x = F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x