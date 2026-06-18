import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The LayerNorm normalizes over the channel dimension (out_channels=64).
# After ConvTranspose3d output is (N, C, D, H, W). nn.LayerNorm(C) applied to
# this tensor would normalize over the LAST dim (W), but the reference's
# semantic intent for "layer_norm over out_channels" requires permuting to
# channels-last. Match the reference behavior literally: LayerNorm normalized
# over the last dim (W). Since the kernel pool's correct version normalizes
# over the last dim of (N,C,D,H,W) which equals W=64=C, this is what we do too.

@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    M, C,
    eps, scaling_factor,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    base = row * C
    x = tl.load(x_ptr + base + cols, mask=mask, other=0.0).to(tl.float32)

    sum_x = tl.sum(tl.where(mask, x, 0.0), axis=0)
    mean = sum_x / C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    g = tl.load(gamma_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    y = xc * rstd * g + b
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = gelu * scaling_factor

    tl.store(out_ptr + base + cols, out, mask=mask)


def ln_gelu_scale(x_channel_last, gamma, beta, eps, scaling_factor):
    C = x_channel_last.shape[-1]
    M = x_channel_last.numel() // C
    out = torch.empty_like(x_channel_last)
    BLOCK_C = triton.next_power_of_2(C)
    num_warps = 4
    if BLOCK_C <= 64:
        num_warps = 2
    _ln_gelu_scale_kernel[(M,)](
        x_channel_last, gamma, beta, out,
        M, C, eps, scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


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
        x = x.contiguous()
        out = ln_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias, self.eps, self.scaling_factor)
        return out