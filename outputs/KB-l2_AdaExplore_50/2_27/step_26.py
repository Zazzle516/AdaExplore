import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Mega-kernel: fused Conv3d + HardSwish + GroupNorm + spatial mean
# One program per B. Emits all OC channels in a single pass.
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
    OC_C: tl.constexpr,
    CHANNELS_PER_GROUP: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    eps: tl.constexpr,
):
    pid_b = tl.program_id(0)

    oc_range = tl.arange(0, OC_C)
    g_range = tl.arange(0, NUM_GROUPS)

    if HAS_BIAS:
        bias_vec = tl.load(b_ptr + oc_range)
    else:
        bias_vec = tl.zeros((OC_C,), dtype=tl.float32)

    # Per-channel running sums [OC]
    c_sums = tl.zeros((OC_C,), dtype=tl.float32)
    # Per-group running sumsq/sum [NUM_GROUPS]
    g_sums = tl.zeros((NUM_GROUPS,), dtype=tl.float32)
    g_sumsq = tl.zeros((NUM_GROUPS,), dtype=tl.float32)

    g_idx_oc = oc_range // CHANNELS_PER_GROUP  # [OC]
    # one-hot [OC, NUM_GROUPS]
    onehot = (g_idx_oc[:, None] == g_range[None, :]).to(tl.float32)

    x_batch_base = pid_b * IC * D * H * W

    for s_start in range(0, OS, BLOCK_S):
        s_offs = s_start + tl.arange(0, BLOCK_S)
        s_mask = s_offs < OS

        od = s_offs // (OH * OW)
        rem = s_offs - od * (OH * OW)
        oh = rem // OW
        ow = rem - oh * OW

        # acc [OC, BLOCK_S]
        acc = tl.zeros((OC_C, BLOCK_S), dtype=tl.float32)

        for k in tl.static_range(0, K):
            ic = k // (KD * KH * KW)
            kr = k - ic * (KD * KH * KW)
            kd = kr // (KH * KW)
            kr2 = kr - kd * (KH * KW)
            kh = kr2 // KW
            kw = kr2 - kh * KW

            in_d = od + kd
            in_h = oh + kh
            in_w = ow + kw
            in_offs = x_batch_base + ic * (D * H * W) + in_d * (H * W) + in_h * W + in_w
            x_val = tl.load(x_ptr + in_offs, mask=s_mask, other=0.0)  # [BLOCK_S]

            w_col = tl.load(w_ptr + oc_range * K + k)  # [OC]
            acc += w_col[:, None] * x_val[None, :]

        acc += bias_vec[:, None]

        # HardSwish
        t = acc + 3.0
        t = tl.minimum(tl.maximum(t, 0.0), 6.0)
        y = acc * t * (1.0 / 6.0)
        y = tl.where(s_mask[None, :], y, 0.0)

        c_sum_tile = tl.sum(y, axis=1)        # [OC]
        c_sumsq_tile = tl.sum(y * y, axis=1)  # [OC]

        c_sums += c_sum_tile

        # group reductions via one-hot matmul-like sum
        g_sums += tl.sum(c_sum_tile[:, None] * onehot, axis=0)
        g_sumsq += tl.sum(c_sumsq_tile[:, None] * onehot, axis=0)

    N_g = CHANNELS_PER_GROUP * OS
    g_mean = g_sums / N_g
    g_var = g_sumsq / N_g - g_mean * g_mean
    g_invstd = 1.0 / tl.sqrt(g_var + eps)

    c_means = c_sums / OS

    # gather per-channel mean/invstd via one-hot
    mean_per_c = tl.sum(onehot * g_mean[None, :], axis=1)
    invstd_per_c = tl.sum(onehot * g_invstd[None, :], axis=1)

    gamma = tl.load(gamma_ptr + oc_range)
    beta = tl.load(beta_ptr + oc_range)
    out_v = gamma * (c_means - mean_per_c) * invstd_per_c + beta
    tl.store(out_ptr + pid_b * OC + oc_range, out_v)


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
        K = IC * KD * KH * KW

        BLOCK_S = 2048

        grid = (B,)
        fused_conv_hs_gn_mean_kernel[grid](
            x, weight, bias, gamma, beta, out,
            B, IC, D, H, W,
            OC, OD, OH, OW, OS,
            KD=KD, KH=KH, KW=KW,
            IC_C=IC,
            OC_C=OC,
            CHANNELS_PER_GROUP=channels_per_group,
            NUM_GROUPS=self.num_groups,
            BLOCK_S=BLOCK_S,
            K=K,
            HAS_BIAS=has_b,
            eps=self.eps,
            num_warps=8,
            num_stages=3,
        )
        return out