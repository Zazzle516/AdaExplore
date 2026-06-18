import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def layernorm_gelu_scale_kernel(
    x_ptr, out_ptr, weight_ptr, bias_ptr,
    M, C,
    eps, scaling_factor,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / C
    xm = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xm * xm, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    y = xm * rstd * w + b
    # GELU (exact via erf)
    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    g = g * scaling_factor
    
    tl.store(out_ptr + row * C + offs, g, mask=mask)


def fused_ln_gelu_scale(x, weight, bias, eps, scaling_factor):
    # x shape: (N, C, D, H, W) - normalize over C dimension
    # Permute to (N, D, H, W, C) for contiguous C
    N, C, D, H, W = x.shape
    x_perm = x.permute(0, 2, 3, 4, 1).contiguous()
    M = N * D * H * W
    out = torch.empty_like(x_perm)
    
    BLOCK_C = triton.next_power_of_2(C)
    
    layernorm_gelu_scale_kernel[(M,)](
        x_perm, out, weight, bias,
        M, C,
        eps, scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    
    return out.permute(0, 4, 1, 2, 3).contiguous()


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = scaling_factor
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_ln_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias, self.eps, self.scaling_factor)
        return x