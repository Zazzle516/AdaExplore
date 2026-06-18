import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def stats_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr,
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    m = tl.load(mult_ptr + c)
    base = n * C * S + c * S

    sum1 = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum2 = tl.zeros([BLOCK_S], dtype=tl.float32)
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        y = x * m
        y = tl.where(mask, y, 0.0)
        sum1 += y
        sum2 += y * y

    s1 = tl.sum(sum1, axis=0)
    s2 = tl.sum(sum2, axis=0)
    mean = s1 / S
    var = s2 / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.jit
def apply_and_max_kernel(
    x_ptr, mult_ptr, mean_ptr, invstd_ptr, out_ptr,
    N, C, S,
    clamp_min, clamp_max,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    NEG_INF = -float('inf')
    max_val = tl.full([BLOCK_S], NEG_INF, dtype=tl.float32)

    for c_i in tl.static_range(0, C_CONST):
        if c_i < C:
            m = tl.load(mult_ptr + c_i)
            mu = tl.load(mean_ptr + pid_n * C + c_i)
            iv = tl.load(invstd_ptr + pid_n * C + c_i)
            x = tl.load(x_ptr + pid_n * C * S + c_i * S + s_offs, mask=s_mask, other=0.0)
            y = x * m
            y = (y - mu) * iv
            y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
            y = y * m
            max_val = tl.maximum(max_val, y)

    tl.store(out_ptr + pid_n * S + s_offs, max_val, mask=s_mask)


@triton.jit
def fused_stats_apply_max_kernel(
    x_ptr, mult_ptr, out_ptr,
    N, C, S,
    clamp_min, clamp_max, eps,
    BLOCK_S: tl.constexpr,
    C_CONST: tl.constexpr,
):
    # One program per N. Loads multiplier and computes mean/var per channel,
    # then applies norm+clamp+mult+max-over-C in second pass over S.
    # Requires storing intermediate per-channel mean/invstd in shared via tl arrays.
    pid_n = tl.program_id(0)

    c_offs = tl.arange(0, C_CONST)
    c_mask = c_offs < C
    mult = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)  # [C_CONST]

    # Compute per-channel sum and sumsq
    sum1 = tl.zeros([C_CONST], dtype=tl.float32)
    sum2 = tl.zeros([C_CONST], dtype=tl.float32)

    base_n = pid_n * C * S
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        # Load [C_CONST, BLOCK_S]
        ptrs = x_ptr + base_n + c_offs[:, None] * S + s_offs[None, :]
        load_mask = c_mask[:, None] & s_mask[None, :]
        x = tl.load(ptrs, mask=load_mask, other=0.0)
        y = x * mult[:, None]
        y = tl.where(load_mask, y, 0.0)
        sum1 += tl.sum(y, axis=1)
        sum2 += tl.sum(y * y, axis=1)

    mean = sum1 / S
    var = sum2 / S - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    # Second pass: apply norm + clamp + mult + max-over-C, write output
    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        ptrs = x_ptr + base_n + c_offs[:, None] * S + s_offs[None, :]
        load_mask = c_mask[:, None] & s_mask[None, :]
        x = tl.load(ptrs, mask=load_mask, other=0.0)
        y = x * mult[:, None]
        y = (y - mean[:, None]) * invstd[:, None]
        y = tl.minimum(tl.maximum(y, clamp_min), clamp_max)
        y = y * mult[:, None]
        # mask out invalid channels with -inf so they don't affect max
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

        out = torch.empty((N, S), device=x.device, dtype=torch.float32)

        # C_CONST must be power of 2 >= C
        C_CONST = 1
        while C_CONST < C:
            C_CONST *= 2

        BLOCK_S = 1024
        fused_stats_apply_max_kernel[(N,)](
            x_flat, mult_flat, out,
            N, C, S,
            self.clamp_min, self.clamp_max, 1e-5,
            BLOCK_S=BLOCK_S,
            C_CONST=C_CONST,
            num_warps=8,
        )

        return out.view(N, D, H, W)