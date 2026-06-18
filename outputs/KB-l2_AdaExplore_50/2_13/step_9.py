import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _post_kernel(
    x_ptr,        # (B, C, H, W) - already mean-pooled
    bias_ptr,     # (C,)
    out_ptr,      # (B, C, H, W)
    B, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    # pid iterates over (B * H * W)
    HW = H * W
    b = pid // HW
    hw = pid % HW
    
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    
    # x layout: (B, C, H, W) -> index = b*C*HW + c*HW + hw
    x_ptrs = x_ptr + b * C * HW + offs_c * HW + hw
    bias_ptrs = bias_ptr + offs_c
    
    x = tl.load(x_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    
    v = x + bias
    v = tl.where(mask_c, v, -float('inf'))
    
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s
    
    # tanh via exp
    # tanh(x) = (exp(2x)-1)/(exp(2x)+1)
    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    
    out = t * scaling_factor
    
    out_ptrs = out_ptr + b * C * HW + offs_c * HW + hw
    tl.store(out_ptrs, out, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels
    
    def forward(self, x):
        x = self.conv_transpose(x)                # (B, C, D, H, W)
        x = x.mean(dim=2)                          # (B, C, H, W)
        x = x.contiguous()
        
        B, C, H, W = x.shape
        out = torch.empty_like(x)
        
        bias_flat = self.bias.view(-1).contiguous()
        
        # Pick BLOCK_C as next power of 2 >= C
        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2
        
        grid = (B * H * W,)
        _post_kernel[grid](
            x, bias_flat, out,
            B, C, H, W,
            float(self.scaling_factor),
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        
        # Restore the keepdim=True D=1 dimension
        return out.unsqueeze(2)