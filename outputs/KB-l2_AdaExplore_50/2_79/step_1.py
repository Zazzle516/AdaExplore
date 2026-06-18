import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr, mult_ptr, out_ptr,
    N, C, S,
    clamp_min, clamp_max,
    eps,
    BLOCK_S: tl.constexpr,
):
    # one program per (n, c) -> computes mean/var, applies pipeline, writes per-element max contribution
    # actually we'll do: one program per (n) and loop over c, then take max across c in a second pass.
    # Simpler: 2 kernels. First kernel: normalize+clamp+mul per (n,c). Second: reduce max over c.
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    base = pid_n * C * S + pid_c * S
    m = tl.load(mult_ptr + pid_c)  # multiplier scalar for this channel

    # First pass: compute sum and sum of squares of (x * m)
    sum_val = 0.0
    sumsq_val = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum_val += tl.sum(y, axis=0)
        sumsq_val += tl.sum(y * y, axis=0)

    mean = sum_val / S
    var = sumsq_val / S - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Second pass: normalize, clamp, multiply by m, store
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = (y - mean) * inv_std
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
        y = y * m
        tl.store(out_ptr + base + offs, y, mask=mask)


@triton.jit
def max_reduce_c_kernel(
    x_ptr, out_ptr,
    N, C, S,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    mask = s_offs < S

    neg_inf = float('-inf')
    acc = tl.full((BLOCK_S,), neg_inf, dtype=tl.float32)

    for c in range(0, C):
        ptr = x_ptr + pid_n * C * S + c * S + s_offs
        v = tl.load(ptr, mask=mask, other=neg_inf)
        acc = tl.maximum(acc, v)

    tl.store(out_ptr + pid_n * S + s_offs, acc, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(N, C, S)
        mult_flat = self.multiplier.contiguous().view(-1)

        intermediate = torch.empty_like(x_flat)

        BLOCK_S = 1024
        grid = (N, C)
        fused_norm_clamp_max_kernel[grid](
            x_flat, mult_flat, intermediate,
            N, C, S,
            self.clamp_min, self.clamp_max,
            1e-5,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        out = torch.empty((N, D, H, W), device=x.device, dtype=x.dtype)
        out_flat = out.view(N, S)

        BLOCK_S2 = 256
        grid2 = (N, (S + BLOCK_S2 - 1) // BLOCK_S2)
        max_reduce_c_kernel[grid2](
            intermediate, out_flat,
            N, C, S,
            BLOCK_S=BLOCK_S2,
            num_warps=4,
        )

        return out