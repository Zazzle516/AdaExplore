import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_kernel(
    x_ptr,        # [N, IC, H_in, W_in]
    w_ptr,        # [OC, IC, KH, KW]
    b_ptr,        # [OC]
    out_ptr,      # [N, OC, H_out, W_out]
    N, IC, H_in, W_in,
    OC, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    OC_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # [BLOCK_S]
    s_mask = s_offs < (H_out * W_out)
    oh = s_offs // W_out
    ow = s_offs % W_out

    # Accumulator: [OC_C, BLOCK_S]
    acc = tl.zeros([OC_C, BLOCK_S], dtype=tl.float32)

    x_batch_ptr = x_ptr + pid_n * IC * H_in * W_in

    for kh in tl.static_range(0, KH):
        for kw in tl.static_range(0, KW):
            ih = oh + kh  # valid input row
            iw = ow + kw  # valid input col
            in_off = ih * W_in + iw  # [BLOCK_S]
            for ic in tl.static_range(0, IC_C):
                x_off = x_batch_ptr + ic * H_in * W_in + in_off
                x_val = tl.load(x_off, mask=s_mask, other=0.0)  # [BLOCK_S]
                # weights: w[oc, ic, kh, kw] for oc in [0, OC_C)
                w_off = tl.arange(0, OC_C) * (IC * KH * KW) + ic * KH * KW + kh * KW + kw
                w_val = tl.load(w_ptr + w_off)  # [OC_C]
                acc += w_val[:, None] * x_val[None, :]

    # Add bias
    bias = tl.load(b_ptr + tl.arange(0, OC_C))  # [OC_C]
    acc += bias[:, None]

    # Store: out[pid_n, oc, s_offs]
    oc_offs = tl.arange(0, OC_C)
    out_base = out_ptr + pid_n * OC * H_out * W_out
    out_offs = oc_offs[:, None] * (H_out * W_out) + s_offs[None, :]
    tl.store(out_base + out_offs, acc, mask=s_mask[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def compute_group_stats_kernel(
    conv_ptr,
    mean_ptr,
    invstd_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    CPG: tl.constexpr,
    eps,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    sum_val = tl.zeros([], dtype=tl.float32)
    sumsq_val = tl.zeros([], dtype=tl.float32)

    NUM_TILES = (S + BLOCK_S - 1) // BLOCK_S
    for ci in range(CPG):
        c = pid_g * CPG + ci
        base = conv_ptr + pid_n * C * S + c * S
        for t in range(NUM_TILES):
            cur_s = t * BLOCK_S + tl.arange(0, BLOCK_S)
            cur_mask = cur_s < S
            v = tl.load(base + cur_s, mask=cur_mask, other=0.0).to(tl.float32)
            sum_val += tl.sum(v, axis=0)
            sumsq_val += tl.sum(v * v, axis=0)

    count = CPG * S
    mean = sum_val / count
    var = sumsq_val / count - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * GROUPS + pid_g, mean)
    tl.store(invstd_ptr + pid_n * GROUPS + pid_g, inv_std)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C', 'S'],
)
@triton.jit
def fused_lse_kernel(
    conv_ptr,
    mean_ptr,
    invstd_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    N, C, S,
    GROUPS: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    max_val = tl.full([BLOCK_S], -float('inf'), dtype=tl.float32)
    sum_exp = tl.zeros([BLOCK_S], dtype=tl.float32)

    for c in range(C):
        g = c // CPG
        mean = tl.load(mean_ptr + pid_n * GROUPS + g)
        invstd = tl.load(invstd_ptr + pid_n * GROUPS + g)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        x = tl.load(conv_ptr + pid_n * C * S + c * S + s_offs, mask=s_mask, other=0.0).to(tl.float32)
        norm = (x - mean) * invstd * gamma + beta
        e2 = tl.exp(2.0 * norm)
        t = (e2 - 1.0) / (e2 + 1.0)
        tp3 = t + 3.0
        tp3_clamped = tl.minimum(tl.maximum(tp3, 0.0), 6.0)
        hs = t * tp3_clamped * (1.0 / 6.0)
        res = x + hs

        new_max = tl.maximum(max_val, res)
        sum_exp = sum_exp * tl.exp(max_val - new_max) + tl.exp(res - new_max)
        max_val = new_max

    lse = tl.log(sum_exp) + max_val
    tl.store(out_ptr + pid_n * S + s_offs, lse, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.eps = eps
        self.cpg = out_channels // groups

    def forward(self, x):
        x = x.contiguous()
        N, IC, H_in, W_in = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        S = H_out * W_out

        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        x_conv = torch.empty((N, OC, H_out, W_out), device=x.device, dtype=x.dtype)

        BLOCK_S = 256
        num_s_tiles = (S + BLOCK_S - 1) // BLOCK_S
        grid = (N, num_s_tiles)
        conv_kernel[grid](
            x, w, b, x_conv,
            N, IC, H_in, W_in,
            OC, H_out, W_out,
            KH=KH, KW=KW,
            IC_C=IC,
            OC_C=OC,
            BLOCK_S=BLOCK_S,
            num_warps=8,
            num_stages=2,
        )

        x_conv_flat = x_conv.reshape(N, OC, S)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        mean = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)

        compute_group_stats_kernel[(N, self.groups)](
            x_conv_flat, mean, invstd,
            N, OC, S,
            GROUPS=self.groups,
            CPG=self.cpg,
            eps=self.eps,
        )

        out = torch.empty((N, 1, H_out, W_out), device=x.device, dtype=x_conv.dtype)
        out_flat = out.reshape(N, S)

        grid_lse = lambda META: (N, (S + META['BLOCK_S'] - 1) // META['BLOCK_S'])
        fused_lse_kernel[grid_lse](
            x_conv_flat, mean, invstd, gamma, beta, out_flat,
            N, OC, S,
            GROUPS=self.groups,
            CPG=self.cpg,
        )

        return out