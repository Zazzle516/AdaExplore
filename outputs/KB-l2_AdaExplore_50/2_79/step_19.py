import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_norm_clamp_max_kernel(
    x_ptr,         # [N, C, S] after conv * multiplier
    mult_ptr,      # [C]
    out_ptr,       # [N, S]
    N, C, S,
    clamp_min, clamp_max,
    eps,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, s_block)
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # First, compute mean and var per channel for this n.
    # We need per-channel statistics over all S elements -> need to loop over S separately.
    # Strategy: for each n, do two passes over S to compute mean and var for each channel.
    # But this kernel is per (n, s_block); we need stats over full S. So instead, we compute
    # stats once outside or use a separate kernel.
    pass


@triton.jit
def _stats_kernel(
    x_ptr,        # [N, C, S]
    mean_ptr,     # [N, C]
    invstd_ptr,   # [N, C]
    N, C, S,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # n * C + c
    n = pid // C
    c = pid % C

    base = n * C * S + c * S

    # accumulate sum and sum of squares
    sum_x = tl.zeros([BLOCK_S], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK_S], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        mask = s_offs < S
        x = tl.load(x_ptr + base + s_offs, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.where(mask, x, 0.0)
        sum_x2 += tl.where(mask, x * x, 0.0)

    mean = tl.sum(sum_x) / S
    mean_x2 = tl.sum(sum_x2) / S
    var = mean_x2 - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * C + c, mean)
    tl.store(invstd_ptr + n * C + c, invstd)


@triton.jit
def _fused_norm_clamp_max_kernel(
    x_ptr,        # [N, C, S]
    mean_ptr,     # [N, C]
    invstd_ptr,   # [N, C]
    mult_ptr,     # [C]
    out_ptr,      # [N, S]
    N, C, S,
    clamp_min, clamp_max,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    # load mean, invstd, mult for all C
    mean = tl.load(mean_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)
    invstd = tl.load(invstd_ptr + pid_n * C + c_offs, mask=c_mask, other=0.0)
    mult = tl.load(mult_ptr + c_offs, mask=c_mask, other=0.0)

    # Output: max over c of clamp((x - mean) * invstd, lo, hi) * mult
    # x has shape [BLOCK_C, BLOCK_S]
    neg_inf = float('-inf')
    max_val = tl.full([BLOCK_S], neg_inf, dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs_i = c_start + tl.arange(0, BLOCK_C)
        c_mask_i = c_offs_i < C

        mean_i = tl.load(mean_ptr + pid_n * C + c_offs_i, mask=c_mask_i, other=0.0)
        invstd_i = tl.load(invstd_ptr + pid_n * C + c_offs_i, mask=c_mask_i, other=0.0)
        mult_i = tl.load(mult_ptr + c_offs_i, mask=c_mask_i, other=0.0)

        # x_ptr offsets: pid_n * C * S + c * S + s
        x_offsets = pid_n * C * S + c_offs_i[:, None] * S + s_offs[None, :]
        mask_2d = c_mask_i[:, None] & s_mask[None, :]
        x = tl.load(x_ptr + x_offsets, mask=mask_2d, other=0.0).to(tl.float32)

        # normalize
        normed = (x - mean_i[:, None]) * invstd_i[:, None]
        # clamp
        clamped = tl.minimum(tl.maximum(normed, clamp_min), clamp_max)
        # multiply by mult
        val = clamped * mult_i[:, None]
        # mask out invalid channels with -inf
        val = tl.where(c_mask_i[:, None], val, neg_inf)
        # reduce max over channel dim
        cur_max = tl.max(val, axis=0)
        max_val = tl.maximum(max_val, cur_max)

    tl.store(out_ptr + pid_n * S + s_offs, max_val, mask=s_mask)


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
        # x shape: [N, C, D, H, W]
        # multiply by multiplier (with shape [C, 1, 1, 1])
        mult = self.multiplier.view(-1).contiguous()  # [C]
        N, C, D, H, W = x.shape
        S = D * H * W

        x_mul = x * self.multiplier  # [N, C, D, H, W]
        x_flat = x_mul.contiguous().view(N, C, S)

        mean = torch.empty((N, C), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, C), device=x.device, dtype=torch.float32)

        # Launch stats kernel
        grid_stats = (N * C,)
        BLOCK_S_STATS = 1024
        _stats_kernel[grid_stats](
            x_flat, mean, invstd,
            N, C, S, self.eps,
            BLOCK_S=BLOCK_S_STATS,
        )

        out = torch.empty((N, S), device=x.device, dtype=x.dtype)

        BLOCK_S = 256
        BLOCK_C = 16
        # pick BLOCK_C >= C ideally to avoid loop
        # C = 16 here
        grid = (N, triton.cdiv(S, BLOCK_S))
        _fused_norm_clamp_max_kernel[grid](
            x_flat, mean, invstd, mult, out,
            N, C, S,
            self.clamp_min, self.clamp_max,
            BLOCK_S=BLOCK_S, BLOCK_C=BLOCK_C,
        )

        return out.view(N, D, H, W)