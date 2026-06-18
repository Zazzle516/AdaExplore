import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_pool_avg_kernel(
    x_ptr,         # (N, C, D, H, W) - convT output
    out_ptr,       # (N, C)
    N, C, D, H, W,
    PD: tl.constexpr, PH: tl.constexpr, PW: tl.constexpr,
    MK: tl.constexpr,
    scale: tl.constexpr,
    inv_count: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    total_pooled = PD * PH * PW
    base = (n * C + c) * D * H * W
    
    acc = 0.0
    for off in range(0, total_pooled, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < total_pooled
        
        pd = idx // (PH * PW)
        rem = idx % (PH * PW)
        ph = rem // PW
        pw = rem % PW
        
        d0 = pd * MK
        h0 = ph * MK
        w0 = pw * MK
        
        max_v = tl.full((BLOCK,), -float('inf'), dtype=tl.float32)
        for kd in tl.static_range(0, MK):
            for kh in tl.static_range(0, MK):
                for kw in tl.static_range(0, MK):
                    d_ = d0 + kd
                    h_ = h0 + kh
                    w_ = w0 + kw
                    in_bounds = (d_ < D) & (h_ < H) & (w_ < W) & mask
                    off_x = base + (d_ * H + h_) * W + w_
                    v = tl.load(x_ptr + off_x, mask=in_bounds, other=-float('inf'))
                    max_v = tl.maximum(max_v, v)
        
        max_v = tl.where(mask, max_v, 0.0)
        acc += tl.sum(max_v, axis=0)
    
    mean = acc * inv_count * scale
    mean = tl.minimum(tl.maximum(mean, 0.0), 1.0)
    tl.store(out_ptr + n * C + c, mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scale, maxpool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.scale = scale
        self.maxpool_kernel_size = maxpool_kernel_size

    def forward(self, x):
        x = self.conv_transpose(x)
        
        N, C, D, H, W = x.shape
        MK = self.maxpool_kernel_size
        PD = D // MK
        PH = H // MK
        PW = W // MK
        
        x = x.contiguous()
        out = torch.empty((N, C, 1, 1, 1), device=x.device, dtype=x.dtype)
        
        total_pooled = PD * PH * PW
        if total_pooled <= 256:
            BLOCK = 256
        elif total_pooled <= 512:
            BLOCK = 512
        elif total_pooled <= 1024:
            BLOCK = 1024
        else:
            BLOCK = 2048
        
        inv_count = 1.0 / float(total_pooled)
        grid = (N * C,)
        fused_pool_avg_kernel[grid](
            x, out,
            N, C, D, H, W,
            PD, PH, PW,
            MK,
            float(self.scale),
            inv_count,
            BLOCK=BLOCK,
            num_warps=8,
        )
        return out