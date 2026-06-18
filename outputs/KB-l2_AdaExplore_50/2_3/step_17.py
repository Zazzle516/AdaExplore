import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,        # input from conv_transpose: [N, C, D, H, W]
    out_ptr,      # output after pool+gelu: [N, C, D//2, H//2, W//2]
    gamma_ptr,    # LayerNorm gamma [C]
    beta_ptr,     # LayerNorm beta  [C]
    sum_w,        # scalar sum_weight
    eps,          # LN epsilon
    N, C, D, H, W,
    Do, Ho, Wo,   # output spatial after pool
    BLOCK_C: tl.constexpr,
):
    # one program per (n, do, ho, wo)
    pid = tl.program_id(0)
    # decode
    wo = pid % Wo
    tmp = pid // Wo
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    n = tmp // Do

    # input window starts
    d0 = do * 2
    h0 = ho * 2
    w0 = wo * 2

    c_off = tl.arange(0, BLOCK_C)
    c_mask = c_off < C

    # We need LN per (n, d, h, w) over C, for each of the 8 input positions,
    # then average them, then GELU.
    # Accumulate the post-LN values across the 8 positions.
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    gamma = tl.load(gamma_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)

    DHW = D * H * W
    HW = H * W

    # iterate over 8 positions
    for i in tl.static_range(0, 8):
        dd = d0 + (i // 4)
        hh = h0 + ((i // 2) % 2)
        ww = w0 + (i % 2)
        # base pointer for this (n, :, dd, hh, ww)
        base = n * C * DHW + dd * HW + hh * W + ww
        # stride along C is DHW
        ptrs = x_ptr + base + c_off * DHW
        x = tl.load(ptrs, mask=c_mask, other=0.0).to(tl.float32) + sum_w

        # compute mean and var across C
        x_zero = tl.where(c_mask, x, 0.0)
        mean = tl.sum(x_zero, axis=0) / C
        diff = tl.where(c_mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        y = (x - mean) * rstd * gamma + beta
        acc += y

    avg = acc / 8.0
    # GELU (exact, using erf)
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * avg * (1.0 + tl.math.erf(avg * inv_sqrt2))

    # store output: layout [N, C, Do, Ho, Wo]
    DHWo = Do * Ho * Wo
    HWo = Ho * Wo
    out_base = n * C * DHWo + do * HWo + ho * Wo + wo
    out_ptrs = out_ptr + out_base + c_off * DHWo
    tl.store(out_ptrs, gelu, mask=c_mask)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.pool_kernel_size = pool_kernel_size
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        pk = self.pool_kernel_size
        # ensure even-divisible by pool kernel and pool = 2,2,2
        assert pk == (2, 2, 2) or list(pk) == [2, 2, 2]
        Do, Ho, Wo = D // 2, H // 2, W // 2

        out = torch.empty((N, C, Do, Ho, Wo), device=x.device, dtype=x.dtype)

        x_c = x.contiguous()
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps
        sum_w = float(self.sum_weight.item())

        BLOCK_C = _next_pow2(C)
        grid = (N * Do * Ho * Wo,)
        fused_post_kernel[grid](
            x_c, out, gamma, beta,
            sum_w, eps,
            N, C, D, H, W,
            Do, Ho, Wo,
            BLOCK_C=BLOCK_C,
            num_warps=4,
        )
        return out