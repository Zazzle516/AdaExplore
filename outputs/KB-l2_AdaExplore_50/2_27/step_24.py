import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SP': 64}, num_warps=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=2),
        triton.Config({'BLOCK_SP': 128}, num_warps=4),
        triton.Config({'BLOCK_SP': 256}, num_warps=4),
        triton.Config({'BLOCK_SP': 256}, num_warps=8),
        triton.Config({'BLOCK_SP': 512}, num_warps=8),
    ],
    key=['B', 'IC', 'D', 'H', 'W', 'OD', 'OH', 'OW'],
)
@triton.jit
def conv3d_hardswish_kernel(
    x_ptr,         # [B, IC, D, H, W]
    w_ptr,         # [OC, IC, KD, KH, KW]
    b_ptr,         # [OC]
    out_ptr,       # [B, OC, OD, OH, OW]
    B, IC,
    D, H, W,
    OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    OC: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_SP: tl.constexpr,
):
    # one program per (batch*oc, output_spatial_tile)
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

    # Load bias for this OC
    bias = tl.load(b_ptr + pid_oc).to(tl.float32)
    acc = tl.zeros((BLOCK_SP,), dtype=tl.float32) + bias

    # Compute convolution: sum over IC, KD, KH, KW
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

    # hardswish: x * clamp(x+3, 0, 6) / 6
    t = acc + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    y = acc * t * (1.0 / 6.0)

    out_base = pid_b * OC * SP_TOTAL + pid_oc * SP_TOTAL
    tl.store(out_ptr + out_base + sp_offs, y, mask=sp_mask)


@triton.jit
def fused_gn_mean_kernel(
    x_ptr,           # [B, C, S]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [B, C]
    B, C, S,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    eps: tl.constexpr,
):
    b = tl.program_id(0)
    g = tl.program_id(1)

    base = b * C * S + g * CHANNELS_PER_GROUP * S

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Per-channel sums (CHANNELS_PER_GROUP scalars in registers)
    c_sum0 = tl.zeros((), dtype=tl.float32)
    c_sum1 = tl.zeros((), dtype=tl.float32)
    c_sum2 = tl.zeros((), dtype=tl.float32)
    c_sum3 = tl.zeros((), dtype=tl.float32)

    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        for c_off in tl.static_range(0, CHANNELS_PER_GROUP):
            c_base = base + c_off * S
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            csum = tl.sum(v, axis=0)
            sum_val += csum
            sumsq_val += tl.sum(v * v, axis=0)
            if c_off == 0:
                c_sum0 += csum
            if c_off == 1:
                c_sum1 += csum
            if c_off == 2:
                c_sum2 += csum
            if c_off == 3:
                c_sum3 += csum

    N = CHANNELS_PER_GROUP * S
    mean = sum_val / N
    var = sumsq_val / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    inv_S = 1.0 / S

    base_c = g * CHANNELS_PER_GROUP
    out_base = b * C + base_c

    g0 = tl.load(gamma_ptr + base_c + 0).to(tl.float32)
    b0 = tl.load(beta_ptr + base_c + 0).to(tl.float32)
    tl.store(out_ptr + out_base + 0, g0 * (c_sum0 * inv_S - mean) * inv_std + b0)

    g1 = tl.load(gamma_ptr + base_c + 1).to(tl.float32)
    b1 = tl.load(beta_ptr + base_c + 1).to(tl.float32)
    tl.store(out_ptr + out_base + 1, g1 * (c_sum1 * inv_S - mean) * inv_std + b1)

    g2 = tl.load(gamma_ptr + base_c + 2).to(tl.float32)
    b2 = tl.load(beta_ptr + base_c + 2).to(tl.float32)
    tl.store(out_ptr + out_base + 2, g2 * (c_sum2 * inv_S - mean) * inv_std + b2)

    g3 = tl.load(gamma_ptr + base_c + 3).to(tl.float32)
    b3 = tl.load(beta_ptr + base_c + 3).to(tl.float32)
    tl.store(out_ptr + out_base + 3, g3 * (c_sum3 * inv_S - mean) * inv_std + b3)


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

        grid = lambda META: (B * OC, (S + META['BLOCK_SP'] - 1) // META['BLOCK_SP'])
        conv3d_hardswish_kernel[grid](
            x, weight, bias, conv_out,
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

        BLOCK_S = 1024
        if S < 1024:
            BLOCK_S = triton.next_power_of_2(S)
            if BLOCK_S < 64:
                BLOCK_S = 64

        x_flat = conv_out.view(B, OC, S)
        grid2 = (B, self.num_groups)
        fused_gn_mean_kernel[grid2](
            x_flat, gamma, beta, out,
            B, OC, S,
            CHANNELS_PER_GROUP=channels_per_group,
            NUM_GROUPS=self.num_groups,
            BLOCK_S=BLOCK_S,
            eps=self.eps,
            num_warps=4,
        )
        return out