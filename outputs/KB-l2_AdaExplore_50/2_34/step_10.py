import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Note: LayerNorm with normalized_shape=out_channels normalizes over the LAST
# dim of x (which is W' in the (N,C,D,H,W) tensor). This is mathematically
# what the reference does (W' happens to equal out_channels=64 here).
# We keep the same semantics: reduce over the last dim of size C=W'=64.

@triton.jit
def _ln_gelu_scale_kernel(
    x_ptr, y_ptr, gamma_ptr, beta_ptr,
    N, C,
    eps, scale,
    ROWS_PER_PROG: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_PROG
    offs = tl.arange(0, BLOCK_C)
    mask_c = offs < C

    g = tl.load(gamma_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)
    b = tl.load(beta_ptr + offs, mask=mask_c, other=0.0).to(tl.float32)

    inv_C = 1.0 / C
    inv_sqrt2 = 0.70710678118654752440

    for i in tl.static_range(0, ROWS_PER_PROG):
        row = row_start + i
        row_mask = row < N
        ptr = x_ptr + row * C + offs
        x = tl.load(ptr, mask=mask_c & row_mask, other=0.0).to(tl.float32)

        mean = tl.sum(x, axis=0) * inv_C
        xc = tl.where(mask_c, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) * inv_C
        rstd = tl.rsqrt(var + eps)

        y = xc * rstd * g + b
        y_gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))
        out = y_gelu * scale

        tl.store(y_ptr + row * C + offs, out, mask=mask_c & row_mask)


def ln_gelu_scale(x, gamma, beta, eps, scale):
    orig_shape = x.shape
    C = orig_shape[-1]
    x2d = x.reshape(-1, C).contiguous()
    N = x2d.shape[0]
    out = torch.empty_like(x2d)
    BLOCK_C = triton.next_power_of_2(C)
    if BLOCK_C <= 64:
        num_warps = 2
    elif BLOCK_C <= 256:
        num_warps = 4
    else:
        num_warps = 8

    ROWS_PER_PROG = 4
    grid = ((N + ROWS_PER_PROG - 1) // ROWS_PER_PROG,)
    _ln_gelu_scale_kernel[grid](
        x2d, out, gamma, beta,
        N, C, eps, scale,
        ROWS_PER_PROG=ROWS_PER_PROG,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
        num_stages=2,
    )
    return out.reshape(orig_shape)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 bias=True, eps=1e-5, scaling_factor=1.0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels,
                                                  kernel_size, stride=stride,
                                                  padding=padding, bias=bias)
        self.layer_norm = nn.LayerNorm(out_channels, eps=eps)
        self.eps = eps
        self.scaling_factor = float(scaling_factor)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        out = ln_gelu_scale(x, self.layer_norm.weight, self.layer_norm.bias,
                            self.eps, self.scaling_factor)
        return out