import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_post_kernel(
    x_ptr,         # input from conv_transpose: [N, C, D, H, W]
    out_ptr,       # output: [N, C, D//2, H//2, W//2]
    gamma_ptr,     # [C]
    beta_ptr,      # [C]
    sum_weight,    # scalar
    N, C, D, H, W,
    Do, Ho, Wo,    # output spatial after pooling
    eps,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, do, ho, wo)
    pid = tl.program_id(0)
    wo = pid % Wo
    tmp = pid // Wo
    ho = tmp % Ho
    tmp = tmp // Ho
    do = tmp % Do
    n = tmp // Do

    # The 8 spatial input positions for this output pool element.
    # pool kernel = 2, stride = 2 (default).
    d0 = do * 2
    h0 = ho * 2
    w0 = wo * 2

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # Compute sum over 8 positions per channel of (x + sum_weight)
    # We need: for each channel c, the 8 values (x+sum_weight), then
    # apply LayerNorm over channel-dim per position, then average those 8 normalized values, then GELU.

    # Layout: input stride: x[n,c,d,h,w] -> n*C*D*H*W + c*D*H*W + d*H*W + h*W + w
    DHW = D * H * W
    HW = H * W

    base_n = n * C * DHW

    # Accumulator for averaged-after-norm value per channel
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    # iterate over 8 positions
    for i in tl.static_range(0, 8):
        dd = d0 + (i // 4)
        hh = h0 + ((i // 2) % 2)
        ww = w0 + (i % 2)
        # load x[n, :, dd, hh, ww] across channels
        ptrs = x_ptr + base_n + offs_c * DHW + dd * HW + hh * W + ww
        x = tl.load(ptrs, mask=mask_c, other=0.0).to(tl.float32)
        x = x + sum_weight
        # LayerNorm over channel dim (norm_shape = (C,))
        # compute mean
        x_zero = tl.where(mask_c, x, 0.0)
        mean = tl.sum(x_zero, axis=0) / C
        diff = tl.where(mask_c, x - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / C
        rstd = 1.0 / tl.sqrt(var + eps)
        gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
        beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
        normed = (x - mean) * rstd * gamma + beta
        acc = acc + normed

    # average pool
    pooled = acc / 8.0

    # GELU (exact)
    inv_sqrt2 = 0.70710678118654752440
    g = 0.5 * pooled * (1.0 + tl.erf(pooled * inv_sqrt2))

    # store: out[n, :, do, ho, wo]
    out_DHW = Do * Ho * Wo
    out_HW = Ho * Wo
    out_ptrs = out_ptr + n * C * out_DHW + offs_c * out_DHW + do * out_HW + ho * Wo + wo
    tl.store(out_ptrs, g, mask=mask_c)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.pool_kernel_size = pool_kernel_size
        self.out_channels = out_channels
        # BLOCK_C must cover C
        self.BLOCK_C = triton.next_power_of_2(out_channels)

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, D, H, W = x.shape
        # require pool kernel of 2 on each dim, divisible
        pk = self.pool_kernel_size
        assert pk == (2, 2, 2)
        assert D % 2 == 0 and H % 2 == 0 and W % 2 == 0
        Do, Ho, Wo = D // 2, H // 2, W // 2
        out = torch.empty((N, C, Do, Ho, Wo), dtype=x.dtype, device=x.device)

        x = x.contiguous()
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps
        sw = float(self.sum_weight.item())

        grid = (N * Do * Ho * Wo,)
        fused_post_kernel[grid](
            x, out, gamma, beta, sw,
            N, C, D, H, W,
            Do, Ho, Wo,
            eps,
            BLOCK_C=self.BLOCK_C,
            num_warps=2,
        )
        return out