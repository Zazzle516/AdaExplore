import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# LayerNorm with norm_shape=(64,) on tensor (N, C=64, D, H, W=64): PyTorch normalizes
# over the last dimension whose size matches normalized_shape. Here both C and W are 64,
# but LayerNorm normalizes over the LAST dim(s), i.e., W (size 64).
# We replicate that.


@triton.jit
def ln_pool_gelu_kernel(
    x_ptr, gamma_ptr, beta_ptr, out_ptr,
    N, C, D, H, W,
    Dp, Hp, Wp,
    eps,
    stride_n, stride_c, stride_d, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_d, out_stride_h, out_stride_w,
    BLOCK_W: tl.constexpr,
    W_RUNTIME: tl.constexpr,
    Wp_RUNTIME: tl.constexpr,
):
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

    base = n * stride_n + c * stride_c

    inv_W = 1.0 / W_RUNTIME

    pool_accum = tl.zeros([BLOCK_W], dtype=tl.float32)

    # Process 4 rows: (d0,h0), (d0,h1), (d1,h0), (d1,h1)
    for ddi in tl.static_range(0, 2):
        for dhi in tl.static_range(0, 2):
            d = dp * 2 + ddi
            h = hp * 2 + dhi
            row_ptr = x_ptr + base + d * stride_d + h * stride_h + offs_w * stride_w
            row = tl.load(row_ptr, mask=mask_w, other=0.0).to(tl.float32)
            row_zeroed = tl.where(mask_w, row, 0.0)
            mean = tl.sum(row_zeroed, axis=0) * inv_W
            diff = tl.where(mask_w, row - mean, 0.0)
            var = tl.sum(diff * diff, axis=0) * inv_W
            inv = 1.0 / tl.sqrt(var + eps)
            normed = (row - mean) * inv * gamma + beta
            pool_accum += tl.where(mask_w, normed, 0.0)

    # Pairwise sum along W to do avg pool of size 2.
    # Use shift: shifted = pool_accum where index even pick pool_accum[i+1], else pool_accum[i-1]?
    # Simpler approach: shift pool_accum by one using tl.where with even/odd mask is not trivial.
    # Use the trick: reshape via two masked sums isn't direct. Use tl.where and rearrange:
    # For pair sum at output position wp: pool_accum[2*wp] + pool_accum[2*wp+1].
    # We can compute "odd" tensor: shift left by 1 lane via tl.where + arange comparisons isn't built-in.
    # Use simulated shift via summation: take even-masked values and odd-masked values; then we want a compacted
    # output. Use the fact that even+odd pair sums can be obtained by adding pool_accum to itself shifted by 1
    # then taking only even positions. Triton supports `tl.where` and arithmetic but not lane shift.
    # Use the approach: write pool_accum to an intermediate via two arange tensors and reduce with tl.sum over
    # axis after reshape. We can reshape! pool_accum has shape [BLOCK_W]; reshape to [BLOCK_W//2, 2] and sum axis=1.

    pa2 = tl.reshape(pool_accum, (BLOCK_W // 2, 2))
    pair_sum = tl.sum(pa2, axis=1)  # shape [BLOCK_W//2]

    offs_wp = tl.arange(0, BLOCK_W // 2)
    mask_wp = offs_wp < Wp_RUNTIME

    avg = pair_sum * (1.0 / 8.0)
    # GELU (erf form)
    gelu = 0.5 * avg * (1.0 + tl.erf(avg * 0.7071067811865475))

    out_off = (n * out_stride_n + c * out_stride_c +
               dp * out_stride_d + hp * out_stride_h + offs_wp * out_stride_w)
    tl.store(out_ptr + out_off, gelu, mask=mask_wp)


def ln_pool_gelu(x, gamma, beta, eps):
    N, C, D, H, W = x.shape
    Dp = D // 2
    Hp = H // 2
    Wp = W // 2
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
        num_stages=2,
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
        gamma = self.norm.weight.contiguous()
        beta = self.norm.bias.contiguous()
        eps = self.norm.eps
        x = x.contiguous()

        if (x.shape[-1] == gamma.shape[0] and
            self.pool_kernel_size == (2, 2, 2) and
            x.shape[-1] % 2 == 0 and x.shape[-2] % 2 == 0 and x.shape[-3] % 2 == 0):
            return ln_pool_gelu(x, gamma, beta, eps)
        else:
            x = self.norm(x)
            x = self.avg_pool(x)
            x = self.gelu(x)
            return x