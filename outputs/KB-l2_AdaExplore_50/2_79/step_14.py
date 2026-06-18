import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 4096}, num_warps=8, num_stages=2),
    ],
    key=['C', 'S'],
)
@triton.jit
def fused_post_kernel(
    x_ptr,           # [N, C, S]
    mult_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    clamp_min: tl.constexpr,
    clamp_max: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # n

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C
    m = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)

    base = pid * C * S

    # Pass 1: compute mean and var per channel via single pass over S
    sum_acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    sumsq_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
        full_mask = c_mask[:, None] & s_mask[None, :]
        x = tl.load(ptrs, mask=full_mask, other=0.0)
        y = x * m[:, None]
        y_masked = tl.where(full_mask, y, 0.0)
        sum_acc += tl.sum(y_masked, axis=1)
        sumsq_acc += tl.sum(y_masked * y_masked, axis=1)

    inv_S = 1.0 / S
    mean = sum_acc * inv_S
    var = sumsq_acc * inv_S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: normalize, clamp, mul, max over C
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        ptrs = x_ptr + base + c_offs[:, None] * S + s_offs[None, :]
        full_mask = c_mask[:, None] & s_mask[None, :]
        x = tl.load(ptrs, mask=full_mask, other=0.0)
        y = x * m[:, None]
        y = (y - mean[:, None]) * invstd[:, None]
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
        y = y * m[:, None]
        neg_inf = float("-inf")
        y_masked = tl.where(c_mask[:, None], y, neg_inf)
        out = tl.max(y_masked, axis=0)
        tl.store(out_ptr + pid * S + s_offs, out, mask=s_mask)


def fused_post_conv(x, multiplier, clamp_min, clamp_max, eps=1e-5):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_flat = x.contiguous().view(N, C, S)
    mult_flat = multiplier.contiguous().view(C)

    out = torch.empty((N, S), device=x.device, dtype=torch.float32)

    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N,)
    fused_post_kernel[grid](
        x_flat, mult_flat, out,
        N, C, S,
        clamp_min=float(clamp_min),
        clamp_max=float(clamp_max),
        eps=float(eps),
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