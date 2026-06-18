import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_maxpool_mean_clamp_kernel(
    x_ptr,         # conv output (N, C, D, H, W)
    out_ptr,       # (N, C)
    N, C, D, H, W,
    Dp, Hp, Wp,    # pooled dims = D//2, H//2, W//2
    scale,
    inv_count,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    nc_base = (n * C + c) * D * H * W
    pooled_total = Dp * Hp * Wp
    
    acc = 0.0
    for off in range(0, pooled_total, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < pooled_total
        pw = idx % Wp
        tmp = idx // Wp
        ph = tmp % Hp
        pd = tmp // Hp
        
        d0 = pd * 2
        h0 = ph * 2
        w0 = pw * 2
        
        b000 = nc_base + d0 * (H * W) + h0 * W + w0
        b001 = b000 + 1
        b010 = b000 + W
        b011 = b010 + 1
        b100 = b000 + H * W
        b101 = b100 + 1
        b110 = b100 + W
        b111 = b110 + 1
        
        v0 = tl.load(x_ptr + b000, mask=mask, other=-1e30)
        v1 = tl.load(x_ptr + b001, mask=mask, other=-1e30)
        v2 = tl.load(x_ptr + b010, mask=mask, other=-1e30)
        v3 = tl.load(x_ptr + b011, mask=mask, other=-1e30)
        v4 = tl.load(x_ptr + b100, mask=mask, other=-1e30)
        v5 = tl.load(x_ptr + b101, mask=mask, other=-1e30)
        v6 = tl.load(x_ptr + b110, mask=mask, other=-1e30)
        v7 = tl.load(x_ptr + b111, mask=mask, other=-1e30)
        
        m = tl.maximum(tl.maximum(tl.maximum(v0, v1), tl.maximum(v2, v3)),
                       tl.maximum(tl.maximum(v4, v5), tl.maximum(v6, v7)))
        m = tl.where(mask, m, 0.0)
        acc += tl.sum(m, axis=0)
    
    mean = acc * inv_count * scale
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
        x = x.contiguous()
        
        N, C, D, H, W = x.shape
        k = self.maxpool_kernel_size
        Dp, Hp, Wp = D // k, H // k, W // k
        
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        
        if k == 2:
            pooled_total = Dp * Hp * Wp
            if pooled_total <= 256:
                BLOCK = 256
            elif pooled_total <= 512:
                BLOCK = 512
            elif pooled_total <= 1024:
                BLOCK = 1024
            elif pooled_total <= 2048:
                BLOCK = 2048
            else:
                BLOCK = 4096
            
            inv_count = 1.0 / float(pooled_total)
            grid = (N * C,)
            fused_maxpool_mean_clamp_kernel[grid](
                x, out,
                N, C, D, H, W,
                Dp, Hp, Wp,
                float(self.scale),
                inv_count,
                BLOCK=BLOCK,
                num_warps=8,
                num_stages=3,
            )
        else:
            x = self.maxpool(x)
            x = x * self.scale
            x = self.global_avg_pool(x)
            x = torch.clamp(x, min=self.clamp_min, max=self.clamp_max)
            return x
        return out