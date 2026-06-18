import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_conv_hs_gn_mean_kernel(
    x_ptr,            # (B, IC, D_in, H_in, W_in)
    w_ptr,            # (OC, IC, KD, KH, KW)
    b_ptr,            # (OC,)
    gamma_ptr,        # (OC,)
    beta_ptr,         # (OC,)
    out_ptr,          # (B, OC)
    B, IC,
    D_in, H_in, W_in,
    D_out, H_out, W_out,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    CPG: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # one program per batch
    b = tl.program_id(0)

    S = D_out * H_out * W_out

    # per-channel sum of HardSwish(conv) and sum-of-squares per channel
    oc_offs = tl.arange(0, OC)  # (OC,)
    ch_sum = tl.zeros((OC,), dtype=tl.float32)
    ch_sumsq = tl.zeros((OC,), dtype=tl.float32)

    # Load bias once: (OC,)
    bias = tl.load(b_ptr + oc_offs).to(tl.float32)

    # Pre-load weights: (OC, IC*KD*KH*KW)
    K = IC * KD * KH * KW
    k_offs = tl.arange(0, IC * KD * KH * KW)  # (K,)
    # weights shape (OC, IC, KD, KH, KW) contiguous; stride in flat = K per OC
    w = tl.load(w_ptr + oc_offs[:, None] * K + k_offs[None, :]).to(tl.float32)  # (OC, K)

    # decompose k into (ic, kd, kh, kw)
    ic_idx = k_offs // (KD * KH * KW)
    rem = k_offs % (KD * KH * KW)
    kd_idx = rem // (KH * KW)
    rem2 = rem % (KH * KW)
    kh_idx = rem2 // KW
    kw_idx = rem2 % KW

    s_offs = tl.arange(0, BLOCK_S)
    num_chunks = (S + BLOCK_S - 1) // BLOCK_S

    for chunk in range(0, num_chunks):
        s_cur = chunk * BLOCK_S + s_offs  # (BLOCK_S,)
        mask_s = s_cur < S

        # decompose s into (d_out, h_out, w_out)
        d_out = s_cur // (H_out * W_out)
        rem_s = s_cur % (H_out * W_out)
        h_out = rem_s // W_out
        w_out = rem_s % W_out

        # For each output spatial position, gather IC*KD*KH*KW input values
        # input index: ((b*IC + ic)*D_in + (d_out+kd))*H_in + (h_out+kh)) * W_in + (w_out+kw)
        # Shape: (BLOCK_S, K)
        d_in = d_out[:, None] + kd_idx[None, :]  # (BLOCK_S, K)
        h_in = h_out[:, None] + kh_idx[None, :]
        w_in = w_out[:, None] + kw_idx[None, :]
        ic_full = ic_idx[None, :]  # broadcast

        in_idx = ((b * IC + ic_full) * D_in + d_in) * H_in * W_in + h_in * W_in + w_in
        # mask
        m = mask_s[:, None]
        x_vals = tl.load(x_ptr + in_idx, mask=m, other=0.0).to(tl.float32)  # (BLOCK_S, K)

        # conv: out[s, oc] = sum_k x_vals[s, k] * w[oc, k] + bias[oc]
        # Use tl.dot: (BLOCK_S, K) x (K, OC) -> (BLOCK_S, OC)
        w_t = tl.trans(w)  # (K, OC)
        conv_out = tl.dot(x_vals, w_t)  # (BLOCK_S, OC)
        conv_out = conv_out + bias[None, :]

        # HardSwish
        t = conv_out + 3.0
        t = tl.maximum(t, 0.0)
        t = tl.minimum(t, 6.0)
        hs = conv_out * t * (1.0 / 6.0)

        # zero out invalid spatial positions
        hs = tl.where(mask_s[:, None], hs, 0.0)

        # accumulate per-channel sum and sumsq
        ch_sum += tl.sum(hs, axis=0)
        ch_sumsq += tl.sum(hs * hs, axis=0)

    # Now compute GroupNorm per group
    # Group index for each channel
    g_idx = oc_offs // CPG  # (OC,)
    # per-group sum and sumsq
    # We need to sum ch_sum and ch_sumsq within each group
    # Use a small loop over groups
    s_f = S.to(tl.float32)
    n_per_group = (S * CPG).to(tl.float32)

    gamma = tl.load(gamma_ptr + oc_offs).to(tl.float32)
    beta = tl.load(beta_ptr + oc_offs).to(tl.float32)

    # compute mean per channel (over spatial)
    ch_mean = ch_sum / s_f  # (OC,)

    # For each group, compute mean and var across (CPG channels * S spatial)
    # Do it with a loop over groups
    out_vals = tl.zeros((OC,), dtype=tl.float32)
    for g in range(0, NUM_GROUPS):
        mask_g = g_idx == g
        sum_g = tl.sum(tl.where(mask_g, ch_sum, 0.0))
        sumsq_g = tl.sum(tl.where(mask_g, ch_sumsq, 0.0))
        mean_g = sum_g / n_per_group
        var_g = sumsq_g / n_per_group - mean_g * mean_g
        rstd_g = 1.0 / tl.sqrt(var_g + eps)
        # output for channels in this group: (ch_mean - mean_g) * rstd_g * gamma + beta
        val = (ch_mean - mean_g) * rstd_g * gamma + beta
        out_vals = tl.where(mask_g, val, out_vals)

    tl.store(out_ptr + b * OC + oc_offs, out_vals)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        B, IC, D_in, H_in, W_in = x.shape
        KD = KH = KW = self.kernel_size
        D_out = D_in - KD + 1
        H_out = H_in - KH + 1
        W_out = W_in - KW + 1
        OC = self.out_channels
        CPG = OC // self.num_groups

        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)

        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else torch.zeros(OC, device=x.device, dtype=x.dtype)

        gamma = self.group_norm.weight.contiguous()
        beta = self.group_norm.bias.contiguous()

        S = D_out * H_out * W_out
        # Choose BLOCK_S: power of 2 >= some target
        # S = 13*30*30 = 11700
        BLOCK_S = 256

        grid = (B,)
        fused_conv_hs_gn_mean_kernel[grid](
            x, weight, bias, gamma, beta, out,
            B, IC,
            D_in, H_in, W_in,
            D_out, H_out, W_out,
            OC=OC,
            KD=KD, KH=KH, KW=KW,
            CPG=CPG,
            NUM_GROUPS=self.num_groups,
            eps=self.eps,
            BLOCK_S=BLOCK_S,
            num_warps=4,
            num_stages=2,
        )
        return out