import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused Conv3d + HardSwish kernel.
# Input layout: x[B, IC, D, H, W] (contiguous)
# Weight layout: w[OC, IC, KD, KH, KW] (contiguous)
# Output layout: out[B, OC, OD, OH, OW] (contiguous)
# One program per (B, OC, spatial_tile).
@triton.jit
def conv3d_hardswish_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, IC, D, H, W,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    IC_C: tl.constexpr,
    BLOCK_S: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    OS = OD * OH * OW
    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = s_offs < OS

    # Decode s_offs into (od, oh, ow)
    od = s_offs // (OH * OW)
    rem = s_offs - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    # Initialize accumulator
    acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

    x_batch_base = pid_b * IC * D * H * W
    w_oc_base = pid_oc * IC_C * KD * KH * KW

    # Loop over input channels and kernel positions
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
        bval = tl.load(b_ptr + pid_oc)
        acc += bval

    # HardSwish: x * clamp(x+3, 0, 6) / 6
    t = acc + 3.0
    t = tl.minimum(tl.maximum(t, 0.0), 6.0)
    y = acc * t * (1.0 / 6.0)

    # Store
    out_base = pid_b * OC * OS + pid_oc * OS + s_offs
    tl.store(out_ptr + out_base, y, mask=s_mask)


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

    GROUP_SIZE = CHANNELS_PER_GROUP * S
    base = b * C * S + g * CHANNELS_PER_GROUP * S

    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # accumulate per-channel sums in registers (small CHANNELS_PER_GROUP)
    # We'll do two passes: first pass computes total sum, sumsq and per-channel sums
    # Use static array via separate scalar accumulators won't be easy; do per-channel inline
    for c_off in tl.static_range(0, CHANNELS_PER_GROUP):
        c_base = base + c_off * S
        c_sum = tl.zeros((), dtype=tl.float32)
        c_sumsq = tl.zeros((), dtype=tl.float32)
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0).to(tl.float32)
            c_sum += tl.sum(v, axis=0)
            c_sumsq += tl.sum(v * v, axis=0)
        sum_val += c_sum
        sumsq_val += c_sumsq
        # store channel mean temporarily into out (will overwrite later)
        c_idx = g * CHANNELS_PER_GROUP + c_off
        tl.store(out_ptr + b * C + c_idx, c_sum / S)

    N = GROUP_SIZE
    mean = sum_val / N
    var = sumsq_val / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)

    for c_off in tl.static_range(0, CHANNELS_PER_GROUP):
        c_idx = g * CHANNELS_PER_GROUP + c_off
        gamma = tl.load(gamma_ptr + c_idx).to(tl.float32)
        beta = tl.load(beta_ptr + c_idx).to(tl.float32)
        c_mean = tl.load(out_ptr + b * C + c_idx).to(tl.float32)
        out_val = gamma * (c_mean - mean) * inv_std + beta
        tl.store(out_ptr + b * C + c_idx, out_val)


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

        out_conv = torch.empty((B, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        weight = self.conv.weight.contiguous()
        if self.has_bias and self.conv.bias is not None:
            bias = self.conv.bias.contiguous()
            has_b = True
        else:
            bias = torch.empty(1, device=x.device, dtype=x.dtype)
            has_b = False

        BLOCK_S = 256
        grid = (B, OC, (OS + BLOCK_S - 1) // BLOCK_S)

        conv3d_hardswish_kernel[grid](
            x, weight, bias, out_conv,
            B, IC, D, H, W,
            OC, OD, OH, OW,
            KD=KD, KH=KH, KW=KW,
            IC_C=IC,
            BLOCK_S=BLOCK_S,
            HAS_BIAS=has_b,
            num_warps=4,
        )

        S = OS
        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)
        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()
        channels_per_group = OC // self.num_groups

        BLOCK_S2 = 1024
        if S < 1024:
            BLOCK_S2 = max(64, triton.next_power_of_2(S))

        grid2 = (B, self.num_groups)
        fused_gn_mean_kernel[grid2](
            out_conv.view(B, OC, S), gamma, beta, out,
            B, OC, S,
            CHANNELS_PER_GROUP=channels_per_group,
            NUM_GROUPS=self.num_groups,
            BLOCK_S=BLOCK_S2,
            eps=self.eps,
            num_warps=4,
        )
        return out