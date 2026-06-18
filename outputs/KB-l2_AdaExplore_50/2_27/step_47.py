import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused Conv3D + HardSwish + GroupNorm + spatial mean.
# Strategy: do the conv ourselves with a custom Triton kernel that produces
# (B, C, S) output in (B, S, C) channels-last layout, fuse hardswish, and
# accumulate per-channel sum and per-group sumsq directly in shared registers.
# Then a tiny finalize kernel computes group mean/var and final output (B, C).

# Configuration constants for this specific problem
# in_channels=3, out_channels=16, kernel=4 => K=64, IC*K=192
# Output spatial: D'=13, H'=29, W'=29 => S=10933
# num_groups=4, CPG=4

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
    ],
    key=['S', 'OC'],
)
@triton.jit
def fused_conv_hs_gnreduce_kernel(
    x_ptr,            # (B, IC, D, H, W)
    w_ptr,            # (OC, IC, KD, KH, KW) flattened to (OC, IC*KD*KH*KW)
    b_ptr,            # (OC,)
    sum_ptr,          # (B, C) per-channel sum of hardswish(conv)
    sumsq_ptr,        # (B, G) per-group sumsq
    B, IC, D, H, W,
    OC, KD, KH, KW,
    OD, OH, OW,
    S,                # OD*OH*OW
    G: tl.constexpr,
    CPG: tl.constexpr,
    BLOCK_S: tl.constexpr,
    KVOL: tl.constexpr,    # IC*KD*KH*KW
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)

    s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)  # (BLOCK_S,)
    s_mask = s_offs < S

    # decompose s into (od, oh, ow)
    ow = s_offs % OW
    tmp = s_offs // OW
    oh = tmp % OH
    od = tmp // OH

    # Load weights: (OC, KVOL)
    oc_offs = tl.arange(0, 16)  # OC=16
    k_offs = tl.arange(0, KVOL)  # KVOL=192
    w = tl.load(w_ptr + oc_offs[:, None] * KVOL + k_offs[None, :])  # (16, 192)
    bias = tl.load(b_ptr + oc_offs)  # (16,)

    # Compute im2col gather and matmul.
    # For each sample point s, we need a vector of length KVOL of input values.
    # acc: (BLOCK_S, OC=16)
    acc = tl.zeros((BLOCK_S, 16), dtype=tl.float32)

    # Build input gather: for each k in [0, KVOL),
    #   ic = k // (KD*KH*KW)
    #   r  = k %  (KD*KH*KW)
    #   kd = r // (KH*KW)
    #   kh = (r // KW) % KH
    #   kw = r % KW
    # input index: ((b*IC + ic)*D + (od+kd))*H*W + (oh+kh)*W + (ow+kw)
    KHW = KH * KW
    KDHW = KD * KHW

    # Loop over k explicitly, accumulating x[:,k] outer w[:,k]
    # But to use tl.dot we need to materialize x_tile (BLOCK_S, KVOL).
    # KVOL=192 small: we can do this.

    # Build x_tile via gather
    # k indices broadcast against s indices
    ic = k_offs // KDHW                     # (KVOL,)
    r = k_offs % KDHW
    kd = r // KHW
    kh = (r // KW) % KH
    kw = r % KW

    # Per-s spatial offsets
    # in_d = od + kd, in_h = oh + kh, in_w = ow + kw
    # input idx = b*IC*D*H*W + ic*D*H*W + in_d*H*W + in_h*W + in_w
    HW = H * W
    DHW = D * HW
    base_b = pid_b * IC * DHW

    # shape (BLOCK_S, KVOL)
    in_d = od[:, None] + kd[None, :]
    in_h = oh[:, None] + kh[None, :]
    in_w = ow[:, None] + kw[None, :]
    in_idx = base_b + ic[None, :] * DHW + in_d * HW + in_h * W + in_w

    mask = s_mask[:, None]  # (BLOCK_S, 1) -- KVOL doesn't need mask since k_offs < KVOL is always true
    x_tile = tl.load(x_ptr + in_idx, mask=mask, other=0.0)  # (BLOCK_S, KVOL)

    # GEMM: acc = x_tile @ w.T
    acc = tl.dot(x_tile, tl.trans(w))  # (BLOCK_S, 16)
    acc = acc + bias[None, :]

    # HardSwish
    t = acc + 3.0
    t = tl.maximum(t, 0.0)
    t = tl.minimum(t, 6.0)
    hs = acc * t * (1.0 / 6.0)
    hs = tl.where(s_mask[:, None], hs, 0.0)

    # Per-channel sum: reduce over BLOCK_S
    ch_sum = tl.sum(hs, axis=0)  # (16,)
    # atomic add into sum_ptr[b, :]
    tl.atomic_add(sum_ptr + pid_b * OC + oc_offs, ch_sum)

    # Per-group sumsq: reshape (BLOCK_S, OC) -> (BLOCK_S, G, CPG), sum over BLOCK_S and CPG
    hs_sq = hs * hs
    # Sum over BLOCK_S first => (16,)
    ch_sumsq = tl.sum(hs_sq, axis=0)  # (16,)
    # Now reshape to (G, CPG) and sum over CPG
    ch_sumsq_2d = tl.reshape(ch_sumsq, (G, CPG))
    g_sumsq = tl.sum(ch_sumsq_2d, axis=1)  # (G,)
    g_offs = tl.arange(0, G)
    tl.atomic_add(sumsq_ptr + pid_b * G + g_offs, g_sumsq)


@triton.jit
def finalize_kernel(
    sum_ptr,       # (B, C)
    sumsq_ptr,     # (B, G)
    gamma_ptr,     # (C,)
    beta_ptr,      # (C,)
    out_ptr,       # (B, C)
    B, C, S,
    G: tl.constexpr,
    CPG: tl.constexpr,
    eps: tl.constexpr,
):
    b = tl.program_id(0)

    c_offs = tl.arange(0, 16)  # C=16
    ch_sum = tl.load(sum_ptr + b * C + c_offs)  # (C,)

    # group sum: reshape (G, CPG)
    ch_sum_2d = tl.reshape(ch_sum, (G, CPG))
    g_sum = tl.sum(ch_sum_2d, axis=1)  # (G,)

    g_offs = tl.arange(0, G)
    g_sumsq = tl.load(sumsq_ptr + b * G + g_offs)  # (G,)

    n = (S * CPG).to(tl.float32)
    g_mean = g_sum / n
    g_var = g_sumsq / n - g_mean * g_mean
    g_rstd = 1.0 / tl.sqrt(g_var + eps)

    # broadcast g_mean (G,) -> (G, CPG) -> (C,)
    g_mean_2d = tl.reshape(g_mean, (G, 1))
    g_mean_b = tl.broadcast_to(g_mean_2d, (G, CPG))
    ch_mean_full = tl.reshape(g_mean_b, (G * CPG,))

    g_rstd_2d = tl.reshape(g_rstd, (G, 1))
    g_rstd_b = tl.broadcast_to(g_rstd_2d, (G, CPG))
    ch_rstd_full = tl.reshape(g_rstd_b, (G * CPG,))

    s_f = S.to(tl.float32)
    gamma = tl.load(gamma_ptr + c_offs)
    beta = tl.load(beta_ptr + c_offs)

    out_val = (ch_sum / s_f - ch_mean_full) * ch_rstd_full * gamma + beta
    tl.store(out_ptr + b * C + c_offs, out_val)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups=4, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        if isinstance(kernel_size, int):
            self.kd = self.kh = self.kw = kernel_size
        else:
            self.kd, self.kh, self.kw = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        B, IC, D, H, W = x.shape
        OC = self.out_channels
        KD, KH, KW = self.kd, self.kh, self.kw
        OD = D - KD + 1
        OH = H - KH + 1
        OW = W - KW + 1
        S = OD * OH * OW
        G = self.num_groups
        CPG = OC // G

        x = x.contiguous()
        weight = self.conv.weight.contiguous().view(OC, IC * KD * KH * KW)
        if self.conv.bias is not None:
            bias = self.conv.bias.contiguous()
        else:
            bias = torch.zeros(OC, device=x.device, dtype=x.dtype)

        sum_buf = torch.zeros((B, OC), device=x.device, dtype=torch.float32)
        sumsq_buf = torch.zeros((B, G), device=x.device, dtype=torch.float32)
        out = torch.empty((B, OC), device=x.device, dtype=x.dtype)

        KVOL = IC * KD * KH * KW

        grid = lambda meta: (B, (S + meta['BLOCK_S'] - 1) // meta['BLOCK_S'])
        fused_conv_hs_gnreduce_kernel[grid](
            x, weight, bias,
            sum_buf, sumsq_buf,
            B, IC, D, H, W,
            OC, KD, KH, KW,
            OD, OH, OW, S,
            G=G, CPG=CPG, KVOL=KVOL,
        )

        finalize_kernel[(B,)](
            sum_buf, sumsq_buf,
            self.group_norm.weight, self.group_norm.bias,
            out,
            B, OC, S,
            G=G, CPG=CPG, eps=self.eps,
        )
        return out