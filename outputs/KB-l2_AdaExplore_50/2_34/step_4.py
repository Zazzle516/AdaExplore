import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# LayerNorm normalizes over the channel dimension (out_channels).
# Input layout from conv_transpose is (N, C, D, H, W). We need LN over C.
# We'll permute to channels_last: (N, D, H, W, C) and apply LN over the last dim.

@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, out_ptr, w_ptr, b_ptr,
    M, C,
    eps, scaling_factor,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    
    x_row_ptr = x_ptr + row * C
    x = tl.load(x_row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / C
    xc = tl.where(mask, x - mean, 0.0)
    sum_sq = tl.sum(xc * xc, axis=0)
    var = sum_sq / C
    rstd = 1.0 / tl.sqrt(var + eps)
    
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    y = xc * rstd * w + b
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = gelu * scaling_factor
    
    tl.store(out_ptr + row * C + offs, out, mask=mask)


def fused_ln_gelu_scale_channels_last(x_cl, weight, bias, eps, scaling_factor):
    # x_cl shape: (N, D, H, W, C), contiguous in C
    N, D, H, W, C = x_cl.shape
    M = N * D * H * W
    x_flat = x_cl.reshape(M, C)
    out_flat = torch.empty_like(x_flat)
    
    BLOCK_C = triton.next_power_of_2(C)
    
    grid = (M,)
    _ln_gelu_scale_kernel[grid](
        x_flat, out_flat, weight, bias,
        M, C,
        eps, scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=2,
        num_stages=2,
    )
    
    return out_flat.view(N, D, H, W, C)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, eps=1e-5, scaling_factor=1.0):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.scaling_factor = scaling_factor
        self.eps = eps
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        # x shape: (N, C, D, H, W). Permute to (N, D, H, W, C) channels-last.
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()
        out_cl = fused_ln_gelu_scale_channels_last(
            x_cl,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.eps,
            self.scaling_factor,
        )
        # Permute back to (N, C, D, H, W)
        out = out_cl.permute(0, 4, 1, 2, 3).contiguous()
        return out