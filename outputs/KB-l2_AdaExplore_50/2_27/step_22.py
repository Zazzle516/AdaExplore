import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Mega-kernel: fused Conv3d + HardSwish + GroupNorm + spatial mean
# One program per (B, group). Computes the conv output on the fly,
# applies hardswish, accumulates per-channel sum and per-group sumsq,
# then emits CHANNELS_PER_GROUP outputs in (B, C).
@triton.jit
def fused_conv_hs_gn_mean_kernel(
    x_ptr,           # [B, IC, D, H, W]
    w_ptr,           # [OC, IC, KD, KH, KW]
    b_ptr,           # [OC]
    gamma_ptr,       # [OC]
    beta_ptr,        # [OC]
    out_ptr,         # [B, OC]
    B, IC, D, H, W,
    OC, OD, OH, OW, OS,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_g = tl.program_id(1)

    # per-channel sums (CHANNELS_PER_GROUP)
    c_sums = tl.zeros((CHANNELS_PER_GROUP,), dtype=tl.float32)
    group_sumsq = tl.zeros((), dtype=tl.float32)
    group_sum = tl.zeros((), dtype=tl.float32)

    x_batch_base = pid_b * IC * D * H * W
    oc_start = pid_g * CHANNELS_PER_GROUP

    # iterate over spatial tiles
    for s_start in range(0, OS, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < OS

        od = s_offs // (OH * OW)
        rem = s_offs - od * (OH * OW)
        oh = rem // OW
        ow = rem - oh * OW

        # For each channel in group, compute conv output
        for c_off in tl.static_range(0, CHANNELS_PER_GROUP):
            oc = oc_start + c_off
            w_oc_base = oc * IC_C * KD * KH * KW
            acc = tl.zeros((BLOCK_S,), dtype=tl.float32)
            for ic in tl.static_range(0, IC_C):
                x_ic_base = x_batch_base + ic * D * H * W
                w_ic_base = w_oc_base + ic * KD * KH * KW
                for kd in tl.static_range(0, KD):
                    for kh in tl.static_range(0, KH):
                        for kw in tl.static_range(0, KW):
                            w_val = tl.load(w_ptr + w_ic_base + kd * KH * KW + kh * KW + kw)
                            in_d = od + kd
                            in_h = oh + kh
                            in_w = ow + kw
                            in_offs = x_ic_base + in_d * (H * W) + in_h * W + in_w
                            x_val = tl.load(x_ptr + in_offs, mask=s_mask, other=0.0)
                            acc += x_val * w_val
            if HAS_BIAS:
                bval = tl.load(b_ptr + oc)
                acc += bval

            # HardSwish
            t = acc + 3.0
            t = tl.minimum(tl.maximum(t, 0.0), 6.0)
            y = acc * t * (1.0 / 6.0)
            # mask out invalid positions
            y = tl.where(s_mask, y, 0.0)

            s_sum = tl.sum(y, axis=0)
            s_sumsq = tl.sum(y * y, axis=0)

            # accumulate per channel
            c_mask = tl.arange(0, CHANNELS_PER_GROUP) == c_off
            c_sums += tl.where(c_mask, s_sum, 0.0)
            group_sum += s_sum
            group_sumsq += s_sumsq

    N = CHANNELS_PER_GROUP * OS
    mean = group_sum / N
    var = group_sumsq / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    # emit outputs
    c_means = c_sums / OS
    c_idx_v = tl.arange(0, CHANNELS_PER_GROUP)
    gamma = tl.load(gamma_ptr + oc_start + c_idx_v).to(tl.float32)
    beta = tl.load(beta_ptr + oc_start + c_idx_v).to(tl.float32)
    out_v = gamma * (c_means - mean) * inv_std + beta
    tl.store(out_ptr + pid_b * OC + oc_start + c_idx_v, out_v)


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
        self.has_bias = bias
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        B, IC, D, H, W = x.shape
        KD, KH, KW = self.kernel_size
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        OC = self.out_channels
        OS = OD * OH * OW

        weight = self.conv.weight.contiguous()
        if self.has_bias and self.conv.bias is not None:
            bias = self.conv.bias.contiguous()
            has_b = True
        else:
            bias = torch.empty(1, device=x.device, dtype=x.dtype)
            has_b = False

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)

        channels_per_group = OC // self.num_groups

        BLOCK_S = 512

        grid = (B, self.num_groups)
        fused_conv_hs_gn_mean_kernel[grid](
            x, weight, bias, gamma, beta, out,
            B, IC, D, H, W,
            OC, OD, OH, OW, OS,
            KD=KD, KH=KH, KW=KW,
            IC_C=IC,
            CHANNELS_PER_GROUP=channels_per_group,
            NUM_GROUPS=self.num_groups,
            BLOCK_S=BLOCK_S,
            HAS_BIAS=has_b,
            eps=self.eps,
            num_warps=4,
            num_stages=2,
        )
        return out