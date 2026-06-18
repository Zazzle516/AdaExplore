import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def gn_stats_kernel(
    conv_ptr,        # [N, C, S]
    mean_ptr,        # [N, GROUPS]
    rstd_ptr,        # [N, GROUPS]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    n = tl.program_id(0)
    g = tl.program_id(1)

    c_offs = g * CH_PER_GROUP + tl.arange(0, CH_PER_GROUP)
    sum_val = tl.zeros([CH_PER_GROUP], dtype=tl.float32)
    sumsq_val = tl.zeros([CH_PER_GROUP], dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        s_o = s_start + tl.arange(0, BLOCK_S)
        s_m = s_o < S
        ptrs = conv_ptr + n * (C * S) + c_offs[:, None] * S + s_o[None, :]
        x = tl.load(ptrs, mask=s_m[None, :], other=0.0).to(tl.float32)
        sum_val += tl.sum(x, axis=1)
        sumsq_val += tl.sum(x * x, axis=1)

    total_sum = tl.sum(sum_val, axis=0)
    total_sumsq = tl.sum(sumsq_val, axis=0)
    count = CH_PER_GROUP * S
    mean = total_sum / count
    var = total_sumsq / count - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + n * GROUPS + g, mean)
    tl.store(rstd_ptr + n * GROUPS + g, rstd)


@triton.jit
def fused_post_kernel(
    conv_ptr,        # [N, C, S]
    mean_ptr,        # [N, GROUPS]
    rstd_ptr,        # [N, GROUPS]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [N, S]
    N, C, S,
    GROUPS: tl.constexpr,
    CH_PER_GROUP: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)
    sb = tl.program_id(1)

    s_offs = sb * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < S

    m_run = tl.full([BLOCK_S], -float('inf'), dtype=tl.float32)
    se_run = tl.zeros([BLOCK_S], dtype=tl.float32)

    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        g_idx = c_offs // CH_PER_GROUP

        mean_v = tl.load(mean_ptr + n * GROUPS + g_idx, mask=c_mask, other=0.0)
        rstd_v = tl.load(rstd_ptr + n * GROUPS + g_idx, mask=c_mask, other=0.0)
        gamma_v = tl.load(gamma_ptr + c_offs, mask=c_mask, other=0.0)
        beta_v = tl.load(beta_ptr + c_offs, mask=c_mask, other=0.0)

        ptrs = conv_ptr + n * (C * S) + c_offs[:, None] * S + s_offs[None, :]
        load_mask = c_mask[:, None] & s_mask[None, :]
        v = tl.load(ptrs, mask=load_mask, other=0.0).to(tl.float32)

        scale_c = rstd_v * gamma_v
        normed = (v - mean_v[:, None]) * scale_c[:, None] + beta_v[:, None]
        e2 = tl.exp(2.0 * normed)
        t = (e2 - 1.0) / (e2 + 1.0)
        tp3 = t + 3.0
        clamped = tl.minimum(tl.maximum(tp3, 0.0), 6.0)
        hs = t * clamped * (1.0 / 6.0)
        r = v + hs

        r = tl.where(c_mask[:, None], r, -float('inf'))

        block_max = tl.max(r, axis=0)
        new_max = tl.maximum(m_run, block_max)
        scale = tl.exp(m_run - new_max)
        scale = tl.where(new_max == -float('inf'), 0.0, scale)
        block_sum = tl.sum(tl.exp(r - new_max[None, :]), axis=0)
        se_run = se_run * scale + block_sum
        m_run = new_max

    out = m_run + tl.log(se_run)
    tl.store(out_ptr + n * S + s_offs, out, mask=s_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_S': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_OC': 128, 'BLOCK_S': 128}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OH', 'OW', 'IC', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr,           # [N, IC, IH, IW]
    w_ptr,           # [OC, IC*KH*KW] flattened
    b_ptr,           # [OC]
    y_ptr,           # [N, OC, OH, OW]
    N, IC, IH, IW,
    OC, OH, OW,
    KH: tl.constexpr,
    KW: tl.constexpr,
    K: tl.constexpr,        # IC*KH*KW
    BLOCK_K: tl.constexpr,  # next pow2 of K
    KHW: tl.constexpr,      # KH*KW
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < OH * OW
    oh = s_offs // OW
    ow = s_offs % OW

    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    ic_idx = k_offs // KHW
    kh_idx = (k_offs % KHW) // KW
    kw_idx = k_offs % KW

    w_ptrs = w_ptr + oc_offs[:, None] * K + k_offs[None, :]
    w_tile = tl.load(w_ptrs, mask=oc_mask[:, None] & k_mask[None, :], other=0.0)

    ih = oh[None, :] + kh_idx[:, None]
    iw = ow[None, :] + kw_idx[:, None]
    x_ptrs = x_ptr + pid_n * (IC * IH * IW) + ic_idx[:, None] * (IH * IW) + ih * IW + iw
    x_load_mask = k_mask[:, None] & s_mask[None, :]
    x_tile = tl.load(x_ptrs, mask=x_load_mask, other=0.0)

    acc = tl.dot(w_tile, x_tile, out_dtype=tl.float32)

    b_val = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += b_val[:, None]

    y_ptrs = y_ptr + pid_n * (OC * OH * OW) + oc_offs[:, None] * (OH * OW) + s_offs[None, :]
    store_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(y_ptrs, acc, mask=store_mask)


def triton_conv2d(x, weight, bias):
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    OH = IH - KH + 1
    OW = IW - KW + 1
    S = OH * OW
    K = IC * KH * KW
    BLOCK_K = triton.next_power_of_2(K)
    if BLOCK_K < 16:
        BLOCK_K = 16

    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=torch.float32)
    w_flat = weight.view(OC, K).contiguous()

    grid = lambda META: (N, triton.cdiv(OC, META['BLOCK_OC']), triton.cdiv(S, META['BLOCK_S']))
    conv2d_kernel[grid](
        x, w_flat, bias, y,
        N, IC, IH, IW,
        OC, OH, OW,
        KH=KH, KW=KW,
        K=K, BLOCK_K=BLOCK_K, KHW=KH * KW,
    )
    return y


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.eps = eps
        self.groups = groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv.weight.contiguous()
        b = self.conv.bias.contiguous()
        x_conv = triton_conv2d(x, w, b)
        N, C, H, W = x_conv.shape
        S = H * W
        x_flat = x_conv.view(N, C, S)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        mean = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N, self.groups), device=x.device, dtype=torch.float32)

        CH_PER_GROUP = C // self.groups
        BLOCK_S = 512

        gn_stats_kernel[(N, self.groups)](
            x_flat, mean, rstd,
            N, C, S,
            GROUPS=self.groups,
            CH_PER_GROUP=CH_PER_GROUP,
            BLOCK_S=BLOCK_S,
            eps=self.eps,
            num_warps=4,
        )

        out = torch.empty((N, 1, H, W), device=x.device, dtype=torch.float32)
        out_flat = out.view(N, S)

        BLOCK_S2 = 256
        BLOCK_C = triton.next_power_of_2(C)

        grid = (N, triton.cdiv(S, BLOCK_S2))
        fused_post_kernel[grid](
            x_flat, mean, rstd, gamma, beta, out_flat,
            N, C, S,
            GROUPS=self.groups,
            CH_PER_GROUP=CH_PER_GROUP,
            BLOCK_S=BLOCK_S2,
            BLOCK_C=BLOCK_C,
            num_warps=8,
        )

        return out