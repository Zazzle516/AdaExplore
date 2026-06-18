import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_htanh_mean_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    H_out, W_out,
    inv_count,
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * H * W + c * H * W
    total = H_out * W_out

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for start in range(0, total, BLOCK):
        idx = start + offs
        mask = idx < total
        oh = idx // W_out
        ow = idx % W_out

        ih0 = oh * 2
        iw0 = ow * 2

        row0 = base + ih0 * W + iw0
        row1 = row0 + W

        v00 = tl.load(x_ptr + row0, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + row0 + 1, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + row1, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + row1 + 1, mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
        m = tl.where(mask, m, 0.0)
        acc += m

    s = tl.sum(acc, axis=0)
    mean = s * inv_count
    e2 = tl.exp(2.0 * mean)
    out_val = (e2 - 1.0) / (e2 + 1.0)
    tl.store(out_ptr + pid, out_val)


@triton.jit
def fused_pool_htanh_mean_kernel_split(
    x_ptr,
    partial_ptr,   # [N, C, NSPLIT]
    N, C, H, W,
    H_out, W_out,
    NSPLIT: tl.constexpr,
    HTANH_MIN: tl.constexpr,
    HTANH_MAX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    base = n * C * H * W + c * H * W
    total = H_out * W_out

    # range of pooled positions for this split
    chunk = (total + NSPLIT - 1) // NSPLIT
    start = pid_s * chunk
    end = start + chunk
    if end > total:
        end = total

    offs = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)

    for s_off in range(start, end, BLOCK):
        idx = s_off + offs
        mask = idx < end
        oh = idx // W_out
        ow = idx % W_out

        ih0 = oh * 2
        iw0 = ow * 2

        row0 = base + ih0 * W + iw0
        row1 = row0 + W

        v00 = tl.load(x_ptr + row0, mask=mask, other=-float('inf'))
        v01 = tl.load(x_ptr + row0 + 1, mask=mask, other=-float('inf'))
        v10 = tl.load(x_ptr + row1, mask=mask, other=-float('inf'))
        v11 = tl.load(x_ptr + row1 + 1, mask=mask, other=-float('inf'))

        m = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
        m = tl.minimum(tl.maximum(m, HTANH_MIN), HTANH_MAX)
        m = tl.where(mask, m, 0.0)
        acc += m

    s = tl.sum(acc, axis=0)
    tl.store(partial_ptr + pid_nc * NSPLIT + pid_s, s)


@triton.jit
def reduce_partial_kernel(
    partial_ptr,   # [NC, NSPLIT]
    out_ptr,       # [NC]
    NC,
    NSPLIT: tl.constexpr,
    inv_count,
):
    pid = tl.program_id(0)
    if pid < NC:
        offs = tl.arange(0, NSPLIT)
        vals = tl.load(partial_ptr + pid * NSPLIT + offs)
        s = tl.sum(vals, axis=0)
        mean = s * inv_count
        e2 = tl.exp(2.0 * mean)
        out_val = (e2 - 1.0) / (e2 + 1.0)
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

        if self.maxpool_kernel_size == 2 and self.maxpool_stride == 2 and H % 2 == 0 and W % 2 == 0:
            H_out = H // 2
            W_out = W // 2
            total = H_out * W_out
            inv_count = 1.0 / total
            NC = N * C

            # Use split-K reduction across spatial when total is large
            # For 128x128 = 16384 pooled positions, single program processes them sequentially.
            # Split into NSPLIT chunks for better parallelism.
            NSPLIT = 8
            partial = torch.empty((NC, NSPLIT), device=x.device, dtype=torch.float32)
            out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

            BLOCK = 1024
            grid1 = (NC, NSPLIT)
            fused_pool_htanh_mean_kernel_split[grid1](
                x, partial,
                N, C, H, W,
                H_out, W_out,
                NSPLIT,
                self.hardtanh_min, self.hardtanh_max,
                BLOCK=BLOCK,
                num_warps=4,
                num_stages=3,
            )
            grid2 = (NC,)
            reduce_partial_kernel[grid2](
                partial, out,
                NC,
                NSPLIT,
                inv_count,
                num_warps=1,
            )
            return out
        else:
            x = F.max_pool2d(x, kernel_size=self.maxpool_kernel_size, stride=self.maxpool_stride)
            x = F.hardtanh(x, min_val=self.hardtanh_min, max_val=self.hardtanh_max)
            x = torch.mean(x, dim=(2, 3), keepdim=True)
            x = torch.tanh(x)
            return x