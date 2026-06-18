import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,          # (B, C, S) input from conv
    out_ptr,        # (B, C) output
    gn_weight_ptr,  # (C,)
    gn_bias_ptr,    # (C,)
    B, C, S,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    S_CONST: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    # one program per (batch, group)
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    # channels in this group
    c_start = pid_g * CHANNELS_PER_GROUP
    c_offs = c_start + tl.arange(0, CHANNELS_PER_GROUP)  # (CPG,)

    # total elements in group
    group_numel = CHANNELS_PER_GROUP * S_CONST

    # Two-pass over S: first compute sum and sum of squares of hardswish(x)
    s_offs = tl.arange(0, BLOCK_S)

    sum_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    sumsq_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)

    for s_start in tl.static_range(0, S_CONST, BLOCK_S):
        s_idx = s_start + s_offs  # (BLOCK_S,)
        s_mask = s_idx < S_CONST
        # pointer: x[b, c, s] = x_ptr[b*C*S + c*S + s]
        ptrs = x_ptr + pid_b * C * S + c_offs[:, None] * S + s_idx[None, :]
        mask = s_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        # hardswish: x * relu6(x+3) / 6
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        hs = tl.where(mask, hs, 0.0)
        sum_acc += tl.sum(hs, axis=1)
        sumsq_acc += tl.sum(hs * hs, axis=1)

    # group reduction across channels
    grp_sum = tl.sum(sum_acc, axis=0)
    grp_sumsq = tl.sum(sumsq_acc, axis=0)

    mean = grp_sum / group_numel
    var = grp_sumsq / group_numel - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # load gn affine
    w = tl.load(gn_weight_ptr + c_offs).to(tl.float32)  # (CPG,)
    b_aff = tl.load(gn_bias_ptr + c_offs).to(tl.float32)

    # Second pass: compute normalized*w+b, then mean over S
    out_acc = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    for s_start in tl.static_range(0, S_CONST, BLOCK_S):
        s_idx = s_start + s_offs
        s_mask = s_idx < S_CONST
        ptrs = x_ptr + pid_b * C * S + c_offs[:, None] * S + s_idx[None, :]
        mask = s_mask[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)
        # normalize
        norm = (hs - mean) * rstd
        val = norm * w[:, None] + b_aff[:, None]
        val = tl.where(mask, val, 0.0)
        out_acc += tl.sum(val, axis=1)

    out_mean = out_acc / S_CONST
    # store: out[b, c]
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
        x = self.conv(x)  # (B, C, D, H, W)
        B, C, D, H, W = x.shape
        S = D * H * W
        x_flat = x.reshape(B, C, S).contiguous()
        out = torch.empty((B, C), device=x.device, dtype=x.dtype)

        CHANNELS_PER_GROUP = C // self.num_groups
        # choose BLOCK_S
        # S = 13*29*29 = 10933 for kernel_size=4 with D=16,H=32,W=32 -> 13,29,29
        BLOCK_S = 256

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
            num_warps=4,
        )
        return out