import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def conv_kernel(
    x_ptr,       # [N, IC, IH, IW]
    w_ptr,       # [OC, IC, KH, KW]
    b_ptr,       # [OC]
    out_ptr,     # [N, OC, OH, OW]
    N, IC: tl.constexpr, IH: tl.constexpr, IW: tl.constexpr,
    OC: tl.constexpr, OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_m = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < (OH * OW)
    oh = m_offs // OW
    ow = m_offs % OW

    oc_offs = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)
    oc_mask = oc_offs < OC

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in tl.static_range(0, IC * KH * KW):
        ic = k // (KH * KW)
        rem = k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        ih = oh + kh
        iw = ow + kw

        x_offs = pid_n * IC * IH * IW + ic * IH * IW + ih * IW + iw
        x_vals = tl.load(x_ptr + x_offs, mask=m_mask, other=0.0)

        w_offs = oc_offs * (IC * KH * KW) + ic * (KH * KW) + kh * KW + kw
        w_vals = tl.load(w_ptr + w_offs, mask=oc_mask, other=0.0)

        acc += x_vals[:, None] * w_vals[None, :]

    b_vals = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_vals[None, :]

    out_offs = pid_n * OC * OH * OW + oc_offs[None, :] * (OH * OW) + m_offs[:, None]
    out_mask = m_mask[:, None] & oc_mask[None, :]
    tl.store(out_ptr + out_offs, acc, mask=out_mask)


@triton.jit
def gn_stats_kernel(
    conv_ptr,
    mean_ptr,
    invstd_ptr,
    N, C: tl.constexpr, S: tl.constexpr,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_g = tl.program_id(1)

    c_offs = pid_g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)

    sum_val = tl.zeros((), dtype=tl.float32)
    sum_sq = tl.zeros((), dtype=tl.float32)

    num_tiles = tl.cdiv(S, BLOCK_S)
    for t in range(0, num_tiles):
        s_offs = t * BLOCK_S + tl.arange(0, BLOCK_S)
        s_mask = s_offs < S
        ptrs = pid_n * C * S + c_offs[:, None] * S + s_offs[None, :]
        x = tl.load(conv_ptr + ptrs, mask=s_mask[None, :], other=0.0).to(tl.float32)
        sum_val += tl.sum(x)
        sum_sq += tl.sum(x * x)

    total = CH_PER_GROUP * S
    mean = sum_val / total
    var = sum_sq / total - mean * mean
    invstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid_n * GROUPS + pid_g, mean)
    tl.store(invstd_ptr + pid_n * GROUPS + pid_g, invstd)


@triton.jit
def gn_apply_lse_online_kernel(
    conv_ptr,    # [N, C, S]
    mean_ptr,    # [N, GROUPS]
    invstd_ptr,  # [N, GROUPS]
    gamma_ptr,   # [C]
    beta_ptr,    # [C]
    out_ptr,     # [N, S]
    N, C: tl.constexpr, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    NEG_INF = float('-inf')
    m_cur = tl.full((BLOCK_S,), NEG_INF, dtype=tl.float32)
    s_cur = tl.zeros((BLOCK_S,), dtype=tl.float32)

    for c in tl.static_range(0, C):
        g = c // CH_PER_GROUP
        mean = tl.load(mean_ptr + pid_n * GROUPS + g)
        invstd = tl.load(invstd_ptr + pid_n * GROUPS + g)
        gamma = tl.load(gamma_ptr + c)
        beta = tl.load(beta_ptr + c)

        ptrs = pid_n * C * S + c * S + s_offs
        x = tl.load(conv_ptr + ptrs, mask=s_mask, other=0.0).to(tl.float32)
        norm = (x - mean) * invstd * gamma + beta
        t = tl.extra.cuda.libdevice.tanh(norm)
        hs_in = t + 3.0
        hs_clamped = tl.minimum(tl.maximum(hs_in, 0.0), 6.0)
        hs = t * hs_clamped * (1.0 / 6.0)
        res = x + hs

        m_new = tl.maximum(m_cur, res)
        s_cur = s_cur * tl.exp(m_cur - m_new) + tl.exp(res - m_new)
        m_cur = m_new

    out = tl.log(s_cur) + m_cur
    out_ptrs = pid_n * S + s_offs
    tl.store(out_ptr + out_ptrs, out, mask=s_mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, x):
        N, IC, IH, IW = x.shape
        OC = self.out_channels
        KH = self.kernel_size
        KW = self.kernel_size
        OH = IH - KH + 1
        OW = IW - KW + 1
        S = OH * OW

        x = x.contiguous()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()

        x_conv = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 128
        BLOCK_N = 64
        grid = (N, triton.cdiv(OC, BLOCK_N), triton.cdiv(S, BLOCK_M))
        conv_kernel[grid](
            x, w, b, x_conv,
            N, IC, IH, IW,
            OC, OH, OW,
            KH, KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        x_flat = x_conv.view(N, OC, S)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        ch_per_group = OC // self.groups

        mean = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)
        invstd = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)

        BLOCK_S_STATS = 1024
        gn_stats_kernel[(N, self.groups)](
            x_flat, mean, invstd,
            N, OC, S,
            GROUPS=self.groups,
            CH_PER_GROUP=ch_per_group,
            BLOCK_S=BLOCK_S_STATS,
            eps=float(self.eps),
            num_warps=4,
        )

        out = torch.empty((N, S), device=x.device, dtype=x_conv.dtype)
        BLOCK_S = 256
        grid2 = (N, triton.cdiv(S, BLOCK_S))
        gn_apply_lse_online_kernel[grid2](
            x_flat, mean, invstd, gamma, beta, out,
            N, OC, S,
            GROUPS=self.groups,
            CH_PER_GROUP=ch_per_group,
            BLOCK_S=BLOCK_S,
            num_warps=4,
        )

        return out.view(N, 1, OH, OW)