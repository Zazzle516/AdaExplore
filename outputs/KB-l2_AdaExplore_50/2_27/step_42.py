import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,
    out_ptr,
    gn_weight_ptr,
    gn_bias_ptr,
    B, C, S,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    S_CONST: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    c_start = pid_g * CHANNELS_PER_GROUP
    c_offs = c_start + tl.arange(0, CHANNELS_PER_GROUP)  # (CPG,)

    group_numel = CHANNELS_PER_GROUP * S_CONST

    s_offs = tl.arange(0, BLOCK_S)

    sum_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    sumsq_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)

    base = pid_b * C * S + c_offs[:, None] * S

    # Pass 1: compute mean/var of hardswish output
    for s_start in tl.static_range(0, S_CONST, BLOCK_S):
        s_idx = s_start + s_offs
        s_mask = s_idx < S_CONST
        ptrs = base + s_idx[None, :]
        mask = s_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0)
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        hs = tl.where(mask, hs, 0.0)
        sum_acc += tl.sum(hs, axis=1)
        sumsq_acc += tl.sum(hs * hs, axis=1)

    grp_sum = tl.sum(sum_acc, axis=0)
    grp_sumsq = tl.sum(sumsq_acc, axis=0)

    mean = grp_sum / group_numel
    var = grp_sumsq / group_numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(gn_weight_ptr + c_offs)
    b_aff = tl.load(gn_bias_ptr + c_offs)

    # The final output is mean over S of: ((hs - mean) * rstd) * w + b_aff
    # = w * rstd * (sum_hs/S - mean) + b_aff
    # sum_hs per channel = sum_acc
    mean_hs_per_c = sum_acc / S_CONST
    out_mean = w * rstd * (mean_hs_per_c - mean) + b_aff
    tl.store(out_ptr + pid_b * C + c_offs, out_mean)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        B, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.reshape(B, C, S).contiguous()
        out = torch.empty((B, C), device=x.device, dtype=x.dtype)

        CHANNELS_PER_GROUP = C // self.num_groups
        BLOCK_S = 1024

        grid = (B, self.num_groups)
        fused_post_conv_kernel[grid](
            x_flat, out,
            self.group_norm.weight, self.group_norm.bias,
            B, C, S,
            CHANNELS_PER_GROUP=CHANNELS_PER_GROUP,
            NUM_GROUPS=self.num_groups,
            S_CONST=S,
            BLOCK_S=BLOCK_S,
            eps=self.eps,
            num_warps=8,
        )
        return out