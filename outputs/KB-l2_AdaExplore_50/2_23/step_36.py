import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def gn_mean_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, G, CPG, S,
    eps: tl.constexpr,
    inv_total: tl.constexpr,
    BLOCK: tl.constexpr,
    CPG_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG_C * S
    base = n * (G * CPG_C * S) + g * CPG_C * S

    # Pass 1: compute sum and sum_sq over the whole group
    # AND per-channel sums simultaneously
    sum_v = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    # per-channel sums stored in a [CPG_C] vector
    per_c = tl.zeros([CPG_C], dtype=tl.float32)

    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_v += v
        sum_sq += v * v

    s_v = tl.sum(sum_v, axis=0)
    s_sq = tl.sum(sum_sq, axis=0)
    inv_gs = 1.0 / (CPG_C * S)
    mean = s_v * inv_gs
    var = s_sq * inv_gs - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: per-channel sums (one channel at a time, but reuse to compute contribution)
    # Sum over s of ((v - mean)*rstd*w_c + b_c) = (sv_c - S*mean)*rstd*w_c + S*b_c
    # We compute sv_c for all channels in one fused pass using 2D tile.
    # Use BLOCK tile over S dimension.
    acc_total = tl.zeros([CPG_C], dtype=tl.float32)
    for off in range(0, S, BLOCK):
        idx = off + tl.arange(0, BLOCK)  # [BLOCK]
        mask = idx < S
        # load tile of shape [CPG_C, BLOCK]
        c_off = tl.arange(0, CPG_C)[:, None] * S
        s_off = idx[None, :]
        ptrs = base + c_off + s_off
        m2 = mask[None, :]
        v = tl.load(x_ptr + ptrs, mask=m2, other=0.0)
        acc_total += tl.sum(v, axis=1)

    # acc_total is per-channel sum of length CPG_C
    c_idx = tl.arange(0, CPG_C)
    c_global = g * CPG_C + c_idx
    w = tl.load(w_ptr + c_global)
    b = tl.load(b_ptr + c_global)
    contrib = (acc_total - S * mean) * rstd * w + S * b
    total_contrib = tl.sum(contrib, axis=0)

    tl.atomic_add(out_ptr + n, total_contrib * inv_total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, D, H, W = x.shape
        S = D * H * W
        x = x.contiguous()

        G = self.num_groups
        CPG = C // G
        total = C * S
        inv_total = 1.0 / total

        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        BLOCK = 2048
        grid = (N * G,)
        gn_mean_kernel[grid](
            x, self.group_norm.weight, self.group_norm.bias, out,
            N, G, CPG, S,
            self.eps, inv_total,
            BLOCK=BLOCK, CPG_C=CPG,
            num_warps=8, num_stages=3,
        )
        return out