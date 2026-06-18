import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _fused_reduce_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    scale,
    clamp_min, clamp_max,
    inv_count,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    spatial = D * H * W
    base = (n * C + c) * spatial

    acc = tl.zeros((1,), dtype=tl.float32)
    # iterate
    offs = tl.arange(0, BLOCK)
    total = spatial
    # loop
    s = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, total, BLOCK):
        idx = start + offs
        mask = idx < total
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        s += tl.where(mask, v, 0.0)
    total_sum = tl.sum(s, axis=0)
    mean = total_sum * inv_count * scale
    mean = tl.minimum(tl.maximum(mean, clamp_min), clamp_max)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = float(scale)
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.maxpool_kernel_size = maxpool_kernel_size
        self.clamp_min = 0.0
        self.clamp_max = 1.0
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        N, C, D, H, W = x.shape
        x = x.contiguous()
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        spatial = D * H * W
        inv_count = 1.0 / spatial
        BLOCK = 1024
        if spatial <= 256:
            BLOCK = 256
        elif spatial <= 512:
            BLOCK = 512
        grid = (N * C,)
        _fused_reduce_kernel[grid](
            x, out,
            N, C, D, H, W,
            self.scale,
            self.clamp_min, self.clamp_max,
            inv_count,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out