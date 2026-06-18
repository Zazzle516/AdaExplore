import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# LayerNorm normalizes over out_channels=C=64 (last dim after permute to NDHWC)
# Fused: LayerNorm(C) + GELU + scale, channels-last layout.
@triton.jit
def _ln_gelu_scale_c_kernel(
    x_ptr, out_ptr, w_ptr, b_ptr,
    M, eps, scaling_factor,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    base = row * BLOCK_C
    x = tl.load(x_ptr + base + offs).to(tl.float32)

    sum_x = tl.sum(x, axis=0)
    mean = sum_x / BLOCK_C
    xc = x - mean
    sum_sq = tl.sum(xc * xc, axis=0)
    var = sum_sq / BLOCK_C
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs).to(tl.float32)
    b = tl.load(b_ptr + offs).to(tl.float32)

    y = xc * rstd * w + b
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
    out = gelu * scaling_factor

    tl.store(out_ptr + base + offs, out)


def fused_ln_gelu_scale_channels_last(x_ndhwc, weight, bias, eps, scaling_factor):
    # x_ndhwc shape: (N, D, H, W, C), contiguous. LN over C.
    assert x_ndhwc.is_contiguous()
    N, D, H, W, C = x_ndhwc.shape
    M = N * D * H * W
    out = torch.empty_like(x_ndhwc)
    BLOCK_C = C  # C=64 typically
    grid = (M,)
    _ln_gelu_scale_c_kernel[grid](
        x_ndhwc, out, weight, bias,
        M, eps, scaling_factor,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out


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
        # x: (N, C, D, H, W) -> permute to (N, D, H, W, C) for channels-last LN
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = fused_ln_gelu_scale_channels_last(
            x,
            self.layer_norm.weight,
            self.layer_norm.bias,
            self.eps,
            self.scaling_factor,
        )
        # Back to (N, C, D, H, W)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x