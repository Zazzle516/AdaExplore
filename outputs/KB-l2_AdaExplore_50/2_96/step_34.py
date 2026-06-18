import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _reduce_kernel(
    x_ptr, out_ptr,
    N, C, S,
    scale,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    base = (n * C + c) * S
    
    acc = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    mean = (acc / S) * scale
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
        self.out_channels = out_channels
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.maxpool(x)
        
        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N * C, S)
        out = torch.empty(N, C, 1, 1, 1, device=x.device, dtype=x.dtype)
        
        # Choose BLOCK_S
        BLOCK_S = 256
        if S <= 64:
            BLOCK_S = 64
        elif S <= 128:
            BLOCK_S = 128
        elif S <= 256:
            BLOCK_S = 256
        elif S <= 512:
            BLOCK_S = 512
        else:
            BLOCK_S = 1024
        
        grid = (N * C,)
        _reduce_kernel[grid](
            x_flat, out,
            N, C, S,
            float(self.scale),
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )
        return out