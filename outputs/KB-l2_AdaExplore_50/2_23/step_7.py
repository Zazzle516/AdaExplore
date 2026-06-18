import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC: tl.constexpr, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_s = tl.program_id(2)

    OS = OD * OH * OW
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    s_mask = offs_s < OS

    od = offs_s // (OH * OW)
    rem = offs_s % (OH * OW)
    oh = rem // OW
    ow = rem % OW

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = offs_oc < OC

    acc = tl.zeros((BLOCK_OC, BLOCK_S), dtype=tl.float32)

    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih = oh + kh
                    iw = ow + kw
                    x_off = ((pid_n * IC + ic) * ID + id_) * (IH * IW) + ih * IW + iw
                    x_val = tl.load(x_ptr + x_off, mask=s_mask, other=0.0)

                    w_off = (offs_oc * IC + ic) * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)

                    acc += w_val[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + offs_oc, mask=oc_mask, other=0.0)
    acc += bias[:, None]

    y_off = (pid_n * OC + offs_oc[:, None]) * OS + offs_s[None, :]
    mask = oc_mask[:, None] & s_mask[None, :]
    tl.store(y_ptr + y_off, acc, mask=mask)


@triton.jit
def gn_mean_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, G, CPG: tl.constexpr, S,
    eps,
    inv_total,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_base = (n * G + g) * CPG * S

    # Pass 1: accumulate per-channel sums and sumsq in registers
    ch_sums = tl.zeros((CPG,), dtype=tl.float32)
    sum_val = 0.0
    sumsq_val = 0.0

    for c in tl.static_range(0, CPG):
        c_base = group_base + c * S
        ch_sum_c = 0.0
        ch_sumsq_c = 0.0
        for s_start in range(0, S, BLOCK_S):
            offs = s_start + tl.arange(0, BLOCK_S)
            mask = offs < S
            v = tl.load(x_ptr + c_base + offs, mask=mask, other=0.0)
            ch_sum_c += tl.sum(v, axis=0)
            ch_sumsq_c += tl.sum(v * v, axis=0)
        # write to ch_sums via masked add
        c_mask = tl.arange(0, CPG) == c
        ch_sums = ch_sums + tl.where(c_mask, ch_sum_c, 0.0)
        sum_val += ch_sum_c
        sumsq_val += ch_sumsq_c

    group_size = CPG * S
    mean = sum_val / group_size
    var = sumsq_val / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    # Compute partial sum analytically:
    # partial = sum_c [ w_c * rstd * (ch_sum_c - S*mean) + b_c * S ]
    ch_idx = g * CPG + tl.arange(0, CPG)
    w = tl.load(weight_ptr + ch_idx)
    b = tl.load(bias_ptr + ch_idx)
    contribs = w * rstd * (ch_sums - S * mean) + b * S
    partial = tl.sum(contribs, axis=0)

    tl.atomic_add(out_ptr + n, partial * inv_total)


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
        x = x.contiguous().cuda()
        N, IC, ID, IH, IW = x.shape
        OC = self.out_channels
        K = self.kernel_size
        OD = ID - K + 1
        OH = IH - K + 1
        OW = IW - K + 1
        OS = OD * OH * OW

        y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_OC = 8
        BLOCK_S = 128

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OS, BLOCK_S))
        conv3d_kernel[grid](
            x, self.conv.weight, self.conv.bias, y,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            K, K, K,
            BLOCK_OC, BLOCK_S,
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
            N, G, CPG, S,
            eps, inv_total,
            BLOCK_S=2048,
            num_warps=8,
        )

        return out