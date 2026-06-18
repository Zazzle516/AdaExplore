import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,        # [N, C, S] conv output
    mult_ptr,     # [C]
    out_ptr,      # [N, S] (max over C)
    N, C, S,
    clamp_min,
    clamp_max,
    eps,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    # Load multiplier [C]
    c_offs = tl.arange(0, C_CONST)
    c_mask = c_offs < C
    mult = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)  # [C]

    # For instance norm, we need mean and var per (n, c) over S.
    # We compute that separately. Here we just apply norm result.

    # Strategy: do everything in this kernel for one (n, s-tile).
    # But mean/var per channel requires reducing over all S. So we need
    # a separate pass to compute mean/var, OR compute it here by iterating.
    # Since S can be large (depth*h*w after conv), we use two kernels.
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def compute_stats_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    mean_ptr,     # [N, C]
    invstd_ptr,   # [N, C]
    N, C, S,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # One program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)

    base = n * C * S + c * S

    sum1 = 0.0
    sum2 = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum1 += tl.sum(y, axis=0)
        sum2 += tl.sum(y * y, axis=0)

    mean = sum1 / S
    var = sum2 / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def apply_and_max_kernel(
    x_ptr,        # [N, C, S]
    mult_ptr,     # [C]
    mean_ptr,     # [N, C]
    invstd_ptr,   # [N, C]
    out_ptr,      # [N, S]
    N, C, S,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S  # [BLOCK_S]

    c_offs = tl.arange(0, C_CONST)
    c_mask = c_offs < C  # [C_CONST]

    mult = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)  # [C_CONST]
    mean = tl.load(mean_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)  # [C_CONST]
    invstd = tl.load(invstd_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)  # [C_CONST]

    # Load [C_CONST, BLOCK_S] tile
    x_addrs = pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    tile_mask = c_mask[:, None] & s_mask[None, :]
    x = tl.load(x_ptr + x_addrs, mask=tile_mask, other=0.0)  # [C_CONST, BLOCK_S]

    y = x * mult[:, None]
    y = (y - mean[:, None]) * invstd[:, None]
    y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
    y = y * mult[:, None]

    # Mask out invalid C with -inf so reduction is correct
    NEG_INF = -float('inf')
    y = tl.where(c_mask[:, None], y, NEG_INF)

    max_val = tl.max(y, axis=0)  # [BLOCK_S]

    tl.store(out_ptr + pid_n * S + s_offs, max_val, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)  # [N, C, D, H, W]
        N, C, D, H, W = x.shape
        S = D * H * W

        x_flat = x.contiguous().view(N, C, S)
        mult_flat = self.multiplier.contiguous().view(-1)

        mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

        # C_CONST must be power of 2 >= C
        C_CONST = 1
        while C_CONST < C:
            C_CONST *= 2

        compute_stats_kernel[(N * C,)](
            x_flat, mult_flat, mean, invstd,
            N, C, S, 1e-5,
        )

        out = torch.empty((N, S), device=x.device, dtype=torch.float32)
        grid = lambda META: (N, triton.cdiv(S, META['BLOCK_S']))
        apply_and_max_kernel[grid](
            x_flat, mult_flat, mean, invstd, out,
            N, C, S,
            self.clamp_min, self.clamp_max,
            C_CONST=C_CONST,
        )

        return out.view(N, D, H, W)