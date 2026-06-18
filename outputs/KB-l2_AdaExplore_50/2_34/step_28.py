import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C,
    eps, scale,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C

    x_row = x_ptr + row * C
    x = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)

    inv_C = 1.0 / C
    mean = tl.sum(x, axis=0) * inv_C
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) * inv_C
    rstd = tl.rsqrt(var + eps)

    g = tl.load(gamma_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    xn = xc * rstd
    y = xn * g + b

    # GELU (erf-based exact form)
    inv_sqrt2 = 0.70710678118654752440
    y_gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = y_gelu * scale

    tl.store(y_ptr + row * C + offs, out, mask=mask)


def ln_gelu_scale_contig(x2d, gamma, beta, eps, scale, C):
    N = x2d.shape[0]
    out = torch.empty_like(x2d)
    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C <= 64:
        num_warps = 2
    elif BLOCK_C <= 128:
        num_warps = 4
    elif BLOCK_C <= 512:
        num_warps = 4
    else:
        num_warps = 8
    _ln_gelu_scale_kernel[(N,)](
        x2d, out, gamma, beta,
        N, C, eps, scale,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                  kernel_size, stride=stride,
                                                  padding=padding, bias=bias)
        # Convert conv weights to channels_last_3d memory format
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = float(scaling_factor)
        self.out_channels = out_channels

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)
        # x is channels_last_3d: memory layout is N,D,H,W,C contiguous
        N, C, D, H, W = x.shape
        # View as (N*D*H*W, C) contiguously without copy via permute
        x_nhwc = x.permute(0, 2, 3, 4, 1)  # contiguous due to channels_last_3d
        x2d = x_nhwc.reshape(-1, C)
        out2d = ln_gelu_scale_contig(x2d, self.layer_norm.weight, self.layer_norm.bias,
                                     self.eps, self.scaling_factor, C)
        out = out2d.view(N, D, H, W, C).permute(0, 4, 1, 2, 3)
        return out