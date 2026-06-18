import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['B', 'C', 'S'],
)
@triton.jit
def fused_post_conv_kernel(
    x_ptr,        # (B, C, S) input from conv
    gamma_ptr,    # (C,)
    beta_ptr,     # (C,)
    out_ptr,      # (B, C) output mean
    B, C, S,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
</old_str_1>

<reasoning_2>
Remove BLOCK_S, num_warps, num_stages from the launch since autotune handles them.
</reasoning_2>

<old_str_2>
        # Choose BLOCK_S as a power-of-two close to S; cap to keep shared memory reasonable
        BLOCK_S = 256
        # Pick next power of 2 helper not needed; 256 works well for typical S

        grid = (B, groups)
        fused_post_conv_kernel[grid](
            x_flat, gamma, beta, out,
            B, C, S,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            EPS=self.eps,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )
        return out
</old_str_2>
<new_str_2>
        grid = (B, groups)
        fused_post_conv_kernel[grid](
            x_flat, gamma, beta, out,
            B, C, S,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            EPS=self.eps,
        )
        return out
</new_str_2>

<old_str_1>
<new_str_1>
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['B', 'C', 'S'],
)
@triton.jit
def fused_post_conv_kernel(
    x_ptr,        # (B, C, S) input from conv
    gamma_ptr,    # (C,)
    beta_ptr,     # (C,)
    out_ptr,      # (B, C) output mean
    B, C, S,
    GROUPS: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program per (batch, group)
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    group_size = CHANNELS_PER_GROUP * S
    base = pid_b * C * S + pid_g * group_size

    # First pass: compute sum and sumsq over the group after hardswish
    sum_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    sumsq_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    mean_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)

    c_off = tl.arange(0, CHANNELS_PER_GROUP)  # [Cg]

    for s_start in range(0, S, BLOCK_S):
        s_off = s_start + tl.arange(0, BLOCK_S)  # [BS]
        mask = s_off < S
        # ptrs shape [Cg, BS]
        ptrs = base + c_off[:, None] * S + s_off[None, :]
        x = tl.load(x_ptr + ptrs, mask=mask[None, :], other=0.0).to(tl.float32)
        # hardswish: x * relu6(x+3) / 6
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        hs = tl.where(mask[None, :], hs, 0.0)
        sum_acc += tl.sum(hs, axis=1)
        sumsq_acc += tl.sum(hs * hs, axis=1)
        mean_acc += tl.sum(hs, axis=1)  # we'll reuse sum_acc for mean per channel

    # group statistics
    total = CHANNELS_PER_GROUP * S
    group_sum = tl.sum(sum_acc, axis=0)
    group_sumsq = tl.sum(sumsq_acc, axis=0)
    mean = group_sum / total
    var = group_sumsq / total - mean * mean
    rstd = 1.0 / tl.sqrt(var + EPS)

    # Load gamma/beta for this group's channels
    c_global = pid_g * CHANNELS_PER_GROUP + c_off
    gamma = tl.load(gamma_ptr + c_global).to(tl.float32)
    beta = tl.load(beta_ptr + c_global).to(tl.float32)

    # Per-channel mean after normalization:
    # y_c = mean_over_s( (hs - mean) * rstd * gamma + beta )
    #     = ( (sum_acc[c]/S) - mean ) * rstd * gamma + beta
    per_ch_hs_mean = sum_acc / S
    out_vals = (per_ch_hs_mean - mean) * rstd * gamma + beta

    # Store to out[b, c_global]
    out_ptrs = pid_b * C + c_global
    tl.store(out_ptr + out_ptrs, out_vals)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)  # (B, C, D, H, W)
        B, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.contiguous().view(B, C, S)

        out = torch.empty((B, C), device=x.device, dtype=x.dtype)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        groups = self.num_groups
        channels_per_group = C // groups

        # Choose BLOCK_S as a power-of-two close to S; cap to keep shared memory reasonable
        BLOCK_S = 256
        # Pick next power of 2 helper not needed; 256 works well for typical S

        grid = (B, groups)
        fused_post_conv_kernel[grid](
            x_flat, gamma, beta, out,
            B, C, S,
            GROUPS=groups,
            CHANNELS_PER_GROUP=channels_per_group,
            EPS=self.eps,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )
        return out