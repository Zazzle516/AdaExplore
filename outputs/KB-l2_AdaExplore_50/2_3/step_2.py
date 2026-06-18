import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Note about LayerNorm in the reference:
# norm_shape = (out_channels,) = (64,) but the conv-transpose output is
# (N, C, D, H, W) = (N, 64, D_out, H_out, W_out). PyTorch's LayerNorm with
# normalized_shape=(64,) normalizes over the *last* dimension whose size matches.
# Since W_out = 64 == out_channels here, LN normalizes over the W axis (length 64).
# We replicate that: per (n, c, d, h), reduce over W.


@triton.jit
def ln_pool_gelu_kernel(
    x_ptr,           # input: (N, C, D, H, W) post-conv+sum
    gamma_ptr,       # (W,)
    beta_ptr,        # (W,)
    out_ptr,         # (N, C, Dp, Hp, Wp)
    N, C, D, H, W,
    Dp, Hp, Wp,
    eps,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
    W_RUNTIME: tl.constexpr,
    Wp_RUNTIME: tl.constexpr,
):
    # one program per (n*C, dp, hp): produces 2 d's x 2 h's of LN, then pools 2x2x2 -> Wp outputs along W axis
    pid_nc = tl.program_id(0)
    pid_dp = tl.program_id(1)
    pid_hp = tl.program_id(2)

    n = pid_nc // C
    c = pid_nc % C

    dp = pid_dp
    hp = pid_hp

    offs_w = tl.arange(0, BLOCK_W)
    mask_w = offs_w < W_RUNTIME

    gamma = tl.load(gamma_ptr + offs_w, mask=mask_w, other=0.0)
    beta = tl.load(beta_ptr + offs_w, mask=mask_w, other=0.0)

    # We'll need 2*2 = 4 LN'd rows of length W (for d0,d1 x h0,h1)
    # Compute on the fly and accumulate sum into pooling along W.

    base = n * stride_n + c * stride_c

    # accumulator for pooled row: length BLOCK_W (we'll use pairwise sum across w later)
    pool_accum = tl.zeros([BLOCK_W], dtype=tl.float32)

    for ddi in tl.static_range(0, 2):
        for dhi in tl.static_range(0, 2):
            d = dp * 2 + ddi
            h = hp * 2 + dhi
            row_ptr = x_ptr + base + d * stride_d + h * stride_h + offs_w * stride_w
            row = tl.load(row_ptr, mask=mask_w, other=0.0).to(tl.float32)
            # mean
            row_zeroed = tl.where(mask_w, row, 0.0)
            mean = tl.sum(row_zeroed, axis=0) / W_RUNTIME
            diff = tl.where(mask_w, row - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) / W_RUNTIME
            inv = 1.0 / tl.sqrt(var + eps)
            normed = (row - mean) * inv * gamma + beta
            pool_accum += tl.where(mask_w, normed, 0.0)

    # pool_accum has the sum of 4 LN'd rows along (d,h). Now we need to also sum over w pairs (avg pool 2 along W).
    # divide by 8 to get average over 2x2x2 window.
    # Reshape: pool_accum length BLOCK_W; sum adjacent pairs.
    # Use a shift: even and odd indices.
    even_mask = (offs_w % 2) == 0
    # We'll create indices for pairs: for output position wp in [0, Wp), sum pool_accum[2*wp] + pool_accum[2*wp+1]
    offs_wp = tl.arange(0, BLOCK_W // 2)
    mask_wp = offs_wp < Wp_RUNTIME

    # gather even and odd
    # We can compute via two loads from pool_accum-equivalent: but pool_accum is a register tensor.
    # Use tl.where with shifted version: easier — re-do: build even / odd by indexing offsets.
    # Trick: compute sum of pool_accum[2*i] + pool_accum[2*i+1] using a reshape via masking and reduction.
    # We'll do it manually: shift by 1 lane.
    # Simple approach: store pool_accum to a scratch via two passes - but we don't have scratch.
    # Alternative: create paired sum using arithmetic. Create even tensor: pool_accum where even, 0 else; similarly odd.
    # Then we need to compact even pairs. Easier: write to output one element per program in a final loop using tl.sum over masked windows.

    # Loop over Wp output positions
    for wp in tl.static_range(0, 256):
        if wp < Wp_RUNTIME:
            wmask = (offs_w == (2 * wp)) | (offs_w == (2 * wp + 1))
            s = tl.sum(tl.where(wmask, pool_accum, 0.0), axis=0)
            avg = s / 8.0
            # GELU (exact erf form)
            gelu = 0.5 * avg * (1.0 + tl.erf(avg * 0.7071067811865475))
            out_off = (n * out_stride_n + c * out_stride_c +
                       dp * out_stride_d + hp * out_stride_h + wp * out_stride_w)
            tl.store(out_ptr + out_off, gelu)


def ln_pool_gelu(x, gamma, beta, pool_k, eps):
    N, C, D, H, W = x.shape
    pkD, pkH, pkW = pool_k
    Dp = D // pkD
    Hp = H // pkH
    Wp = W // pkW
    assert pkD == 2 and pkH == 2 and pkW == 2, "kernel hardcoded for 2x2x2 pool"
    out = torch.empty((N, C, Dp, Hp, Wp), device=x.device, dtype=x.dtype)

    BLOCK_W = triton.next_power_of_2(W)
    if BLOCK_W < 16:
        BLOCK_W = 16

    grid = (N * C, Dp, Hp)
    ln_pool_gelu_kernel[grid](
        x, gamma, beta, out,
        N, C, D, H, W,
        Dp, Hp, Wp,
        eps,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3), out.stride(4),
        BLOCK_W=BLOCK_W,
        W_RUNTIME=W,
        Wp_RUNTIME=Wp,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding,
                 sum_weight, norm_shape, pool_kernel_size):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        self.pool_kernel_size = pool_kernel_size
        self.norm_shape = norm_shape

    def forward(self, x):
        x = self.conv_transpose(x)
        x = x + self.sum_weight
        # LayerNorm over last dim (W) since norm_shape=(64,) matches W=64
        # Then 2x2x2 avg pool, then GELU — fused.
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps
        x = x.contiguous()

        # Verify last dim matches norm_shape
        if x.shape[-1] == gamma.shape[0] and self.pool_kernel_size == (2, 2, 2):
            return ln_pool_gelu(x, gamma, beta, self.pool_kernel_size, eps)
        else:
            # fallback
            x = self.norm(x)
            x = self.avg_pool(x)
            x = self.gelu(x)
            return x