import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SP': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SP': 1024}, num_warps=8, num_stages=3),
    ],
    key=['B', 'IC', 'D', 'H', 'W', 'OD', 'OH', 'OW', 'OC'],
)
@triton.jit
def conv3d_hardswish_gn_stats_kernel(
    x_ptr,         # [B, IC, D, H, W]
    w_ptr,         # [OC, IC, KD, KH, KW]
    b_ptr,         # [OC]
    out_ptr,       # [B, OC, OD, OH, OW]
    sum_ptr,       # [B, OC]
    sumsq_ptr,     # [B, OC]
    B, IC,
    D, H, W,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OC: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    pid_boc = tl.program_id(0)
    pid_b = pid_boc // OC
    pid_oc = pid_boc % OC
    pid_sp = tl.program_id(1)

    sp_offs = pid_sp * BLOCK_SP + tl.arange(0, BLOCK_SP)
    OHW = OH * OW
    SP_TOTAL = OD * OH * OW
    sp_mask = sp_offs < SP_TOTAL

    od = sp_offs // OHW
    rem = sp_offs % OHW
    oh = rem // OW
    ow = rem % OW

    bias = tl.load(b_ptr + pid_oc).to(tl.float32)
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32) + bias

    DHW = D * H * W
    HW = H * W
    x_batch_base = pid_b * IC * DHW
    w_oc_base = pid_oc * IC_C * KD * KH * KW

    for ic in tl.static_range(0, IC_C):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    x_idx = x_batch_base + ic * DHW + id_ * HW + ih_ * W + iw_
                    w_idx = w_oc_base + ic * KD * KH * KW + kd * KH * KW + kh * KW + kw
                    xv = tl.load(x_ptr + x_idx, mask=sp_mask, other=0.0).to(tl.float32)
                    wv = tl.load(w_ptr + w_idx).to(tl.float32)
                    acc += xv * wv

    # hardswish
    t = acc + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    y = acc * t * (1.0 / 6.0)

    y_masked = tl.where(sp_mask, y, 0.0)
    s = tl.sum(y_masked, axis=0)
    sq = tl.sum(y_masked * y_masked, axis=0)

    out_base = pid_b * OC * SP_TOTAL + pid_oc * SP_TOTAL
    tl.store(out_ptr + out_base + sp_offs, y, mask=sp_mask)

    # atomic add for partial sums per (b, oc)
    tl.atomic_add(sum_ptr + pid_b * OC + pid_oc, s)
    tl.atomic_add(sumsq_ptr + pid_b * OC + pid_oc, sq)


@triton.jit
def gn_mean_finalize_kernel(
    sum_ptr,       # [B, OC]
    sumsq_ptr,     # [B, OC]
    gamma_ptr,     # [OC]
    beta_ptr,      # [OC]
    out_ptr,       # [B, OC]
    B, OC, S,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    eps: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    c_off = tl.arange(0, CHANNELS_PER_GROUP)
    c_idx = g * CHANNELS_PER_GROUP + c_off

    s_vals = tl.load(sum_ptr + b * OC + c_idx).to(tl.float32)
    sq_vals = tl.load(sumsq_ptr + b * OC + c_idx).to(tl.float32)

    group_sum = tl.sum(s_vals, axis=0)
    group_sq = tl.sum(sq_vals, axis=0)

    N = CHANNELS_PER_GROUP * S
    mean = group_sum / N
    var = group_sq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    inv_S = 1.0 / S

    gamma = tl.load(gamma_ptr + c_idx).to(tl.float32)
    beta = tl.load(beta_ptr + c_idx).to(tl.float32)

    c_mean = s_vals * inv_S
    out_val = gamma * (c_mean - mean) * inv_std + beta
    tl.store(out_ptr + b * OC + c_idx, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = tuple(kernel_size)
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        B, IC, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        OC = self.out_channels
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        S = OD * OH * OW

        weight = self.conv.weight.contiguous()
        if self.conv.bias is not None:
            bias = self.conv.bias.contiguous()
        else:
            bias = torch.zeros(OC, device=x.device, dtype=x.dtype)

        conv_out = torch.empty((B, OC, OD, OH, OW), device=x.device, dtype=x.dtype)
        sum_buf = torch.zeros((B, OC), device=x.device, dtype=torch.float32)
        sumsq_buf = torch.zeros((B, OC), device=x.device, dtype=torch.float32)

        grid = lambda META: (B * OC, (S + META['BLOCK_SP'] - 1) // META['BLOCK_SP'])
        conv3d_hardswish_gn_stats_kernel[grid](
            x, weight, bias, conv_out, sum_buf, sumsq_buf,
            B, IC,
            D, H, W,
            OD, OH, OW,
            KD=KD, KH=KH, KW=KW,
            OC=OC,
            IC_C=IC,
        )

        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        channels_per_group = OC // self.num_groups

        grid2 = (B, self.num_groups)
        gn_mean_finalize_kernel[grid2](
            sum_buf, sumsq_buf, gamma, beta, out,
            B, OC, S,
            CHANNELS_PER_GROUP=channels_per_group,
            NUM_GROUPS=self.num_groups,
            eps=self.eps,
            num_warps=1,
        )
        return out