import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_post_conv_kernel(
    x_ptr,            # (B, C, S) input from conv
    out_ptr,          # (B, C) output
    gamma_ptr,        # (C,)
    beta_ptr,         # (C,)
    B, C, S,
    channels_per_group,
    num_groups,
    eps,
    BLOCK_S: tl.constexpr,
    CPG: tl.constexpr,
):
    # one program per (batch, group)
    b = tl.program_id(0)
    g = tl.program_id(1)

    # group has CPG channels, each with S spatial elements
    c_offs = tl.arange(0, CPG) + g * CPG       # (CPG,)
    s_offs = tl.arange(0, BLOCK_S)             # (BLOCK_S,)

    # Pass 1: compute sum and sum of squares of hardswish(x) over the group
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # iterate over S in chunks
    num_chunks = (S + BLOCK_S - 1) // BLOCK_S
    for chunk in range(0, num_chunks):
        s_cur = chunk * BLOCK_S + s_offs       # (BLOCK_S,)
        mask_s = s_cur < S                      # (BLOCK_S,)

        # ptrs: b * (C*S) + c_offs * S + s_cur
        ptrs = x_ptr + b * (C * S) + c_offs[:, None] * S + s_cur[None, :]
        mask = mask_s[None, :]
        x = tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)

        # hardswish: x * relu6(x+3) / 6
        t = x + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = x * t * (1.0 / 6.0)

        # mask out invalid
        hs = tl.where(mask, hs, 0.0)
        sum_val += tl.sum(hs)
        sumsq_val += tl.sum(hs * hs)

    n = (S * CPG).to(tl.float32)
    mean = sum_val / n
    var = sumsq_val / n - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: compute normalized values, multiply by gamma, add beta,
    # then take mean over S per channel, and write to out (B, C)
    # output[b, c] = mean_s( (hs - mean) * rstd * gamma[c] + beta[c] )
    #             = ( mean_s(hs) - mean ) * rstd * gamma[c] + beta[c]
    # So just compute per-channel mean of hs.
    for ci in range(0, CPG):
        c = g * CPG + ci
        # compute sum of hs over S for this channel
        ch_sum = tl.zeros((), dtype=tl.float32)
        for chunk in range(0, num_chunks):
            s_cur = chunk * BLOCK_S + s_offs
            mask_s = s_cur < S
            ptrs = x_ptr + b * (C * S) + c * S + s_cur
            x = tl.load(ptrs, mask=mask_s, other=0.0).to(tl.float32)
            t = x + 3.0
            t = tl.maximum(t, 0.0)
            t = tl.minimum(t, 6.0)
            hs = x * t * (1.0 / 6.0)
            hs = tl.where(mask_s, hs, 0.0)
            ch_sum += tl.sum(hs)
        ch_mean = ch_sum / S.to(tl.float32)
        gamma = tl.load(gamma_ptr + c).to(tl.float32)
        beta = tl.load(beta_ptr + c).to(tl.float32)
        out_val = (ch_mean - mean) * rstd * gamma + beta
        tl.store(out_ptr + b * C + c, out_val)


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

        cpg = C // self.num_groups
        # choose BLOCK_S
        BLOCK_S = 1024
        # power of two for CPG
        CPG = cpg

        grid = (B, self.num_groups)
        fused_post_conv_kernel[grid](
            x_flat, out,
            self.group_norm.weight, self.group_norm.bias,
            B, C, S,
            cpg, self.num_groups,
            self.eps,
            BLOCK_S=BLOCK_S,
            CPG=CPG,
            num_warps=4,
        )
        return out