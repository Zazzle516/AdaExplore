import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
    IC_C: tl.constexpr,
):
    n = tl.program_id(0)
    oc_block = tl.program_id(1)
    s_block = tl.program_id(2)

    OS = OD * OH * OW
    oc_offs = oc_block * BLOCK_OC + tl.arange(0, BLOCK_OC)
    s_offs = s_block * BLOCK_S + tl.arange(0, BLOCK_S)

    oc_mask = oc_offs < OC
    s_mask = s_offs < OS

    od = s_offs // (OH * OW)
    rem = s_offs % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_S), dtype=tl.float32)

    for ic in tl.static_range(0, IC_C):
        for kd in tl.static_range(0, KD):
            id_ = od + kd
            for kh in tl.static_range(0, KH):
                ih = oh + kh
                for kw in tl.static_range(0, KW):
                    iw = ow + kw
                    x_off = ((n * IC + ic) * ID + id_) * (IH * IW) + ih * IW + iw
                    x_val = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)
                    w_off = ((oc_offs * IC + ic) * KD + kd) * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                    acc += w_val[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + oc_offs, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    y_off = (n * OC + oc_offs[:, None]) * OS + s_offs[None, :]
    out_mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(y_ptr + y_off, acc, mask=out_mask)


@triton.jit
def gn_mean_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, G, CPG: tl.constexpr, S, C,
    eps,
    inv_total,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S

    group_base = n * (C * S) + g * CPG * S

    # Single pass: accumulate per-channel sums (small constant CPG channels)
    # and total sum/sumsq for the group normalization stats.
    sum_val = tl.zeros((), dtype=tl.float32)
    sumsq_val = tl.zeros((), dtype=tl.float32)

    # Per-channel sums stored in a [CPG] tensor would be ideal, but we
    # accumulate into a flat array of registers via an inline loop.
    # We'll do two passes since CPG-sized register array is awkward;
    # but reuse loaded data efficiently.
    # Strategy: store per-channel running sums in a tensor of size [CPG]
    # using a 1D accumulator:
    ch_sum = tl.zeros((CPG,), dtype=tl.float32)

    for c in tl.static_range(0, CPG):
        c_base = group_base + c * S
        c_sum = tl.zeros((), dtype=tl.float32)
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            c_sum += tl.sum(v, axis=0)
            sumsq_val += tl.sum(v * v, axis=0)
        sum_val += c_sum
        # store per-channel sum into ch_sum at index c
        ch_sum = tl.where(tl.arange(0, CPG) == c, c_sum, ch_sum)

    mean = sum_val / group_size
    var = sumsq_val / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Compute output contribution: sum over c of (w_c*rstd*(s_sum_c - S*mean) + S*b_c)
    ch_idx = g * CPG + tl.arange(0, CPG)
    w = tl.load(weight_ptr + ch_idx)
    b = tl.load(bias_ptr + ch_idx)
    contrib = w * rstd * (ch_sum - S * mean) + S * b
    out_acc = tl.sum(contrib, axis=0)

    tl.atomic_add(out_ptr + n, out_acc * inv_total)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1
        OS = OD * OH * OW

        y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 8
        BLOCK_S = 256
        IC_C = IC

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OS, BLOCK_S))
        conv3d_kernel[grid](
            x, self.conv.weight, self.conv.bias, y,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_S=BLOCK_S,
            IC_C=IC_C,
            num_warps=4,
            num_stages=2,
        )

        G = self.num_groups
        CPG = OC // G
        S = OS
        C = OC
        eps = self.group_norm.eps
        total = C * S
        inv_total = 1.0 / total

        out = torch.zeros(N, device=x.device, dtype=x.dtype)
        grid2 = (N * G,)
        gn_mean_kernel[grid2](
            y, self.group_norm.weight, self.group_norm.bias, out,
            N, G, CPG, S, C,
            eps, inv_total,
            BLOCK_S=1024,
            num_warps=4,
        )
        return out