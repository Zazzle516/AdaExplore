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
    gamma_ptr,    # LayerNorm gamma [W]  (norm over last dim)
    beta_ptr,     # LayerNorm beta  [W]
    sum_w,        # scalar sum_weight
    eps,          # LN epsilon
    N, C, D, H, W,
    Do, Ho, Wo,
    BLOCK_W: tl.constexpr,
):
    # one program per (n, c, do, ho) -> produces all wo for this row-pair
    pid = tl.program_id(0)
    ho = pid % Ho
    tmp = pid // Ho
    do = tmp % Do
    tmp = tmp // Do
    c = tmp % C
    n = tmp // C

    d0 = do * 2
    h0 = ho * 2

    w_off = tl.arange(0, BLOCK_W)
    w_mask = w_off < W

    gamma = tl.load(gamma_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)
    beta = tl.load(beta_ptr + w_off, mask=w_mask, other=0.0).to(tl.float32)

    DHW = D * H * W
    HW = H * W
    nc_base = n * C * DHW + c * DHW
    inv_W = 1.0 / W

    # Load and normalize the 4 rows: (d0,h0), (d0,h0+1), (d0+1,h0), (d0+1,h0+1)
    base00 = nc_base + d0 * HW + h0 * W
    base01 = nc_base + d0 * HW + (h0 + 1) * W
    base10 = nc_base + (d0 + 1) * HW + h0 * W
    base11 = nc_base + (d0 + 1) * HW + (h0 + 1) * W

    x00 = tl.load(x_ptr + base00 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    x01 = tl.load(x_ptr + base01 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    x10 = tl.load(x_ptr + base10 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w
    x11 = tl.load(x_ptr + base11 + w_off, mask=w_mask, other=0.0).to(tl.float32) + sum_w

    # normalize each
    def _norm(x):
        return x

    x00z = tl.where(w_mask, x00, 0.0)
    m00 = tl.sum(x00z, axis=0) * inv_W
    d00 = tl.where(w_mask, x00 - m00, 0.0)
    v00 = tl.sum(d00 * d00, axis=0) * inv_W
    r00 = 1.0 / tl.sqrt(v00 + eps)
    y00 = (x00 - m00) * r00 * gamma + beta

    x01z = tl.where(w_mask, x01, 0.0)
    m01 = tl.sum(x01z, axis=0) * inv_W
    d01 = tl.where(w_mask, x01 - m01, 0.0)
    v01 = tl.sum(d01 * d01, axis=0) * inv_W
    r01 = 1.0 / tl.sqrt(v01 + eps)
    y01 = (x01 - m01) * r01 * gamma + beta

    x10z = tl.where(w_mask, x10, 0.0)
    m10 = tl.sum(x10z, axis=0) * inv_W
    d10 = tl.where(w_mask, x10 - m10, 0.0)
    v10 = tl.sum(d10 * d10, axis=0) * inv_W
    r10 = 1.0 / tl.sqrt(v10 + eps)
    y10 = (x10 - m10) * r10 * gamma + beta

    x11z = tl.where(w_mask, x11, 0.0)
    m11 = tl.sum(x11z, axis=0) * inv_W
    d11 = tl.where(w_mask, x11 - m11, 0.0)
    v11 = tl.sum(d11 * d11, axis=0) * inv_W
    r11 = 1.0 / tl.sqrt(v11 + eps)
    y11 = (x11 - m11) * r11 * gamma + beta

    # sum across the 4 rows -> [BLOCK_W]
    s = y00 + y01 + y10 + y11

    # Now for each output wo: avg over (s[2*wo] + s[2*wo+1]) / 8
    # Use shifted load trick: pair-sum
    # We'll iterate over wo and store one element at a time.
    DHWo = Do * Ho * Wo
    HWo = Ho * Wo
    out_base = n * C * DHWo + c * DHWo + do * HWo + ho * Wo
    inv_sqrt2 = 0.7071067811865475

    # For wo in [0, Wo): compute (s[2*wo] + s[2*wo+1]) / 8, gelu, store
    # We rely on Wo being small and use static_range up to a max; but Wo varies.
    # Instead vectorize: build pair sums via even/odd selection.
    # Create offsets for even and odd lanes
    wo_off = tl.arange(0, BLOCK_W // 2)
    wo_mask = wo_off < Wo
    # gather even and odd values from s using shifts: not directly possible.
    # Instead, reshape via masking: even = s where w%2==0, odd = s where w%2==1
    # Sum pairs by computing dot with selection masks is complex; easier: store s temporarily
    # Use a simple approach: scan with even mask and use tl.sum on segments isn't supported.
    # Workaround: compute even and odd via masked extraction using arange comparison:
    even_mask = (w_off % 2 == 0) & w_mask
    odd_mask = (w_off % 2 == 1) & w_mask
    # We need to map even indices 0,2,4,... to 0,1,2,... in BLOCK_W//2
    # Simplest: do two element-wise stores using a loop over wo with scalar extraction is not allowed.
    # Alternative: compute pooled values by re-loading s twice with shifted alignment.
    # Use the trick: pooled[i] = (s[2i] + s[2i+1]) / 8
    # Build via: take s, multiply with even_mask -> scatter-add into half-size? Not in triton easily.
    # Fallback: manual loop over wo using mask trick — accumulate via sum with one-hot mask.
    # Since BLOCK_W is small (next_pow2(W)=64), loop over Wo with static_range over BLOCK_W//2.

    for i in tl.static_range(0, BLOCK_W // 2):
        in_range = i < Wo
        sel0 = (w_off == (2 * i))
        sel1 = (w_off == (2 * i + 1))
        v0 = tl.sum(tl.where(sel0, s, 0.0), axis=0)
        v1 = tl.sum(tl.where(sel1, s, 0.0), axis=0)
        avg = (v0 + v1) * 0.125
        gelu = 0.5 * avg * (1.0 + tl.math.erf(avg * inv_sqrt2))
        tl.store(out_ptr + out_base + i, gelu, mask=in_range)


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
        grid = (N * C * Do * Ho,)
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