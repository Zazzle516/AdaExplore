import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_gn_mean_kernel(
    x_ptr,           # [B, C, S] hardswish(conv) output
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [B, C]
    B, C, S,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    # one program per (batch, group)
    b = tl.program_id(0)
    g = tl.program_id(1)

    GROUP_SIZE = CHANNELS_PER_GROUP * S  # elements in this group
    base = b * C * S + g * CHANNELS_PER_GROUP * S

    # Compute mean and var over the group
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # iterate over channels in group, and over spatial in BLOCK_S chunks
    for c_off in tl.static_range(0, CHANNELS_PER_GROUP):
        c_base = base + c_off * S
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(v, axis=0)
            sumsq_val += tl.sum(v * v, axis=0)

    N = GROUP_SIZE
    mean = sum_val / N
    var = sumsq_val / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # Now for each channel: compute mean over spatial of normalized*gamma + beta
    # mean_c = gamma * (mean_x_c - mean) * inv_std + beta
    for c_off in tl.static_range(0, CHANNELS_PER_GROUP):
        c_base = base + c_off * S
        c_idx = g * CHANNELS_PER_GROUP + c_off
        gamma = tl.load(gamma_ptr + c_idx).to(tl.float32)
        beta = tl.load(beta_ptr + c_idx).to(tl.float32)

        c_sum = tl.zeros((), dtype=tl.float32)
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            c_sum += tl.sum(v, axis=0)

        c_mean = c_sum / S
        out_val = gamma * (c_mean - mean) * inv_std + beta
        tl.store(out_ptr + b * C + c_idx, out_val)


@triton.jit
def hardswish_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # hardswish: x * relu6(x+3) / 6
    t = x + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    y = x * t * (1.0 / 6.0)
    tl.store(out_ptr + offs, y, mask=mask)


def triton_hardswish(x):
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 1024
    grid = ((n + BLOCK - 1) // BLOCK,)
    hardswish_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=4)
    return out


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

        # hardswish (fused triton)
        x = triton_hardswish(x.contiguous())

        # fused groupnorm + spatial mean
        x_flat = x.view(B, C, S)
        out = torch.empty((B, C), device=x.device, dtype=x.dtype)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        channels_per_group = C // self.num_groups

        # pick BLOCK_S
        BLOCK_S = 1024
        if S < 1024:
            BLOCK_S = triton.next_power_of_2(S)
            if BLOCK_S < 64:
                BLOCK_S = 64

        grid = (B, self.num_groups)
        fused_gn_mean_kernel[grid](
            x_flat, gamma, beta, out,
            B, C, S,
            CHANNELS_PER_GROUP=channels_per_group,
            NUM_GROUPS=self.num_groups,
            BLOCK_S=BLOCK_S,
            eps=self.eps,
            num_warps=4,
        )
        return out