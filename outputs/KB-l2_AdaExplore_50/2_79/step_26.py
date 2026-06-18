import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=16, num_stages=2),
    ],
    key=['S', 'C'],
)
@triton.jit
def _fused_all_kernel(
    x_ptr,        # [N, C, S] raw conv output
    mult_ptr,     # [C]
    out_ptr,      # [N, S]
    S,
    clamp_min: tl.constexpr, clamp_max: tl.constexpr,
    eps: tl.constexpr,
    C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)

    base = pid_n * C * S

    c_offs = tl.arange(0, C)
    mult = tl.load(mult_ptr + c_offs)  # [C]
    mult2 = mult * mult

    # Pass 1: compute mean and var per channel of raw x, then scale by mult.
    # mean(x*m) = m * mean(x), var(x*m) = m^2 * var(x)
    sum_x = tl.zeros([C], dtype=tl.float32)
    sum_x2 = tl.zeros([C], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        offs = base + c_offs[:, None] * S + s_offs[None, :]
        x_raw = tl.load(x_ptr + offs, mask=s_mask[None, :], other=0.0).to(tl.float32)
        sum_x += tl.sum(x_raw, axis=1)
        sum_x2 += tl.sum(x_raw * x_raw, axis=1)

    inv_S = 1.0 / S
    mean_x = sum_x * inv_S
    var_x = sum_x2 * inv_S - mean_x * mean_x
    mean = mean_x * mult
    var = var_x * mult2
    invstd = 1.0 / tl.sqrt(var + eps)

    # Precompute coefficients: normed = (x*m - mean) * invstd = x*(m*invstd) - mean*invstd
    a = mult * invstd       # coefficient on x_raw
    b = mean * invstd       # constant subtracted
    # final val = clamp(a * x_raw - b, lo, hi) * mult

    # Pass 2: normalize, clamp, multiply, reduce max over C, write [N, S]
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        offs = base + c_offs[:, None] * S + s_offs[None, :]
        x_raw = tl.load(x_ptr + offs, mask=s_mask[None, :], other=0.0).to(tl.float32)
        normed = x_raw * a[:, None] - b[:, None]
        clamped = tl.minimum(tl.maximum(normed, clamp_min), clamp_max)
        val = clamped * mult[:, None]
        out_val = tl.max(val, axis=0)
        tl.store(out_ptr + pid_n * S + s_offs, out_val, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        S = D * H * W

        x_flat = x.contiguous().view(N, C, S)
        mult = self.multiplier.view(-1).contiguous()

        out = torch.empty((N, S), device=x.device, dtype=x.dtype)

        grid = (N,)
        _fused_all_kernel[grid](
            x_flat, mult, out,
            S,
            self.clamp_min, self.clamp_max,
            self.eps,
            C=C,
        )

        return out.view(N, D, H, W)