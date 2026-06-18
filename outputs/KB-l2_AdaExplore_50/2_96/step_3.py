import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_reduce_kernel(
    x_ptr,
    out_ptr,
    N, C, D, H, W,
    scale,
    inv_count,
    BLOCK: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    total = D * H * W
    base = (n * C + c) * total
    
    acc = 0.0
    for off in range(0, total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc += tl.sum(v, axis=0)
    
    mean = acc * inv_count * scale
    # clamp
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool = nn.MaxPool3d(kernel_size=maxpool_kernel_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.clamp_min = 0
        self.clamp_max = 1
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        
        N, C, D, H, W = x.shape
        x = x.contiguous()
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        
        total = D * H * W
        BLOCK = 1024
        # find suitable block - use power of 2 >= reasonable
        if total <= 256:
            BLOCK = 256
        elif total <= 512:
            BLOCK = 512
        elif total <= 1024:
            BLOCK = 1024
        else:
            BLOCK = 1024
        
        inv_count = 1.0 / float(total)
        grid = (N * C,)
        fused_reduce_kernel[grid](
            x, out,
            N, C, D, H, W,
            float(self.scale),
            inv_count,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out