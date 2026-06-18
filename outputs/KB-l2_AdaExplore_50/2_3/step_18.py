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
    gamma_ptr,    # LayerNorm gamma [W]
    beta_ptr,     # LayerNorm beta  [W]
    sum_w,        # scalar sum_weight
    eps,          # LN epsilon
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, c, do, ho, wo) output position
    pid = tl.program_id(0)
    wo = pid % Wo
    tmp = pid // Wo
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    tmp = tmp // Do
    c = tmp % C
    n = tmp // C

    d0 = do * 2
    h0 = ho * 2
    w0 = wo * 2

    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < W

    gamma = tl.load(gamma_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

    DHW = D * H * W
    HW = H * W
    inv_W = 1.0 / W

    # accumulator over 8 positions
    acc = 0.0

    base_nc = n * C * DHW + c * DHW

    # iterate over 4 (dd,hh) pairs, each loads a full W row
    for i in tl.static_range(0, 4):
        dd = d0 + (i // 2)
        hh = h0 + (i % 2)
        row_base = base_nc + dd * HW + hh * W
        ptrs = x_ptr + row_base + w_off
        x = tl.load(ptrs, mask=w_mask, other=0.0).to(tl.float32) + sum_w

        x_zero = tl.where(w_mask, x, 0.0)
        mean = tl.sum(x_zero, axis=0) * inv_W
        diff = tl.where(w_mask, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) * inv_W
        rstd = 1.0 / tl.sqrt(var + eps)
        y = (x - mean) * rstd * gamma + beta

        # extract values at w0 and w0+1
        sel = tl.where((w_off == w0) | (w_off == (w0 + 1)), y, 0.0)
        acc += tl.sum(sel, axis=0)

    avg = acc / 8.0
    inv_sqrt2 = 0.7071067811865475
    gelu = 0.5 * avg * (1.0 + tl.math.erf(avg * inv_sqrt2))

    DHWo = Do * Ho * Wo
    HWo = Ho * Wo
    out_idx = n * C * DHWo + c * DHWo + do * HWo + ho * Wo + wo
    tl.store(out_ptr + out_idx, gelu)


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

        BLOCK_W = _next_pow2(W)
        grid = (N * C * Do * Ho * Wo,)
        fused_post_kernel[grid](
            x_c, out, gamma, beta,
            sum_w, eps,
            N, C, D, H, W,
            Do, Ho, Wo,
            BLOCK_W=BLOCK_W,
            num_warps=4,
            num_stages=2,
        )
        return out