import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_S': 8192}, num_warps=16, num_stages=2),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def stats_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    mean_ptr,        # [N, C]
    invstd_ptr,      # [N, C]
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)

    sum_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
    sumsq_acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    base = n * C * S + c * S
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        x = tl.where(mask, x, 0.0)
        sum_acc += x
        sumsq_acc += x * x

    s_red = tl.sum(sum_acc, axis=0)
    sq_red = tl.sum(sumsq_acc, axis=0)
    inv_S = 1.0 / S
    mean_x = s_red * inv_S
    var_x = sq_red * inv_S - mean_x * mean_x
    mean_y = m * mean_x
    var_y = m * m * var_x
    invstd = 1.0 / tl.sqrt(var_y + eps)
    tl.store(mean_ptr + n * C + c, mean_y)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=16, num_stages=2),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def apply_norm_clamp_max_kernel(
    x_ptr,
    mult_ptr,
    mean_ptr,
    invstd_ptr,
    out_ptr,
    N, C, S,
    clamp_min,
    clamp_max,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)

    m = tl.load(mult_ptr + c_offs)
    mean = tl.load(mean_ptr + pid_n * C + c_offs)
    invstd = tl.load(invstd_ptr + pid_n * C + c_offs)

    # mean and invstd are already in y-space; y = x*m, normalized = (y - mean)*invstd
    # = x * (m*invstd) - mean*invstd
    scale = m * invstd
    shift = -mean * invstd

    x_ptrs = x_ptr + pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
    x = tl.load(x_ptrs, mask=s_mask[None, :], other=0.0)

    y = x * scale[:, None] + shift[:, None]
    y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
    y = y * m[:, None]

    out = tl.max(y, axis=0)

    tl.store(out_ptr + pid_n * S + s_offs, out, mask=s_mask)


def fused_post_conv(x, multiplier, clamp_min, clamp_max, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N, C, S)
    mult_flat = multiplier.contiguous().view(C)

    mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
    invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

    stats_kernel[(N * C,)](
        x_flat, mult_flat, mean, invstd,
        N, C, S, eps,
    )

    out = torch.empty((N, S), device=x.device, dtype=torch.float32)
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = lambda META: (N, (S + META['BLOCK_S'] - 1) // META['BLOCK_S'])
    apply_norm_clamp_max_kernel[grid](
        x_flat, mult_flat, mean, invstd, out,
        N, C, S, clamp_min, clamp_max,
        BLOCK_C=BLOCK_C,
    )

    return out.view(N, D, H, W)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x):
        x = self.conv(x)
        return fused_post_conv(x, self.multiplier, self.clamp_min, self.clamp_max, eps=1e-5)