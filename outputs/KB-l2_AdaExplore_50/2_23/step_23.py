import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Conv3d kernel: implicit im2col GEMM
# Output layout: (N, OC, OD, OH, OW) contiguous
# -----------------------------------------------------------------------------
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,   # OC tile
    BLOCK_N: tl.constexpr,   # spatial tile
):
    pid_n = tl.program_id(0)            # batch
    pid_m = tl.program_id(1)            # OC tile
    pid_s = tl.program_id(2)            # spatial tile

    out_spatial = OD * OH * OW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)         # OC indices
    offs_s = pid_s * BLOCK_N + tl.arange(0, BLOCK_N)         # spatial indices

    mask_m = offs_m < OC
    mask_s = offs_s < out_spatial

    # decode spatial -> (od, oh, ow)
    od = offs_s // (OH * OW)
    rem = offs_s - od * (OH * OW)
    oh = rem // OW
    ow = rem - oh * OW

    K = IC * KD * KH * KW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # iterate over K = IC * KD * KH * KW
    # Unroll over kd, kh, kw, ic
    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw

                    # weight index: (oc, ic, kd, kh, kw)
                    w_idx = offs_m * (IC * KD * KH * KW) + ic * (KD * KH * KW) + kd * (KH * KW) + kh * KW + kw
                    w_val = tl.load(w_ptr + w_idx, mask=mask_m, other=0.0)  # (BLOCK_M,)

                    # input index: (n, ic, id_, ih_, iw_)
                    x_idx = pid_n * (IC * ID * IH * IW) + ic * (ID * IH * IW) + id_ * (IH * IW) + ih_ * IW + iw_
                    x_val = tl.load(x_ptr + x_idx, mask=mask_s, other=0.0)  # (BLOCK_N,)

                    acc += w_val[:, None] * x_val[None, :]

    # add bias
    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None]

    # store: out[n, oc, od, oh, ow]
    out_idx = pid_n * (OC * out_spatial) + offs_m[:, None] * out_spatial + offs_s[None, :]
    tl.store(y_ptr + out_idx, acc, mask=mask_m[:, None] & mask_s[None, :])


# -----------------------------------------------------------------------------
# Fused GroupNorm + mean reduction kernel
# One program per (N, G). Each program normalizes its group and accumulates
# the sum of normalized outputs into a partial buffer of shape (N, G).
# Then we sum the (N, G) buffer over G and divide by (C * S) to get the mean.
#
# This avoids materializing the normalized tensor when only the mean is needed.
# -----------------------------------------------------------------------------
@triton.jit
def gn_mean_kernel(
    x_ptr, w_ptr, b_ptr, partial_ptr,
    N, G, CPG, S, eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G
    group_size = CPG * S
    base = n * (G * CPG * S) + g * CPG * S

    # Pass 1: mean / var
    sum_v = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        sum_v += v
        sum_sq += v * v
    mean_v = tl.sum(sum_v, axis=0) / group_size
    var = tl.sum(sum_sq, axis=0) / group_size - mean_v * mean_v
    rstd = 1.0 / tl.sqrt(var + eps)

    # Pass 2: compute sum of normalized values for this group
    # y = (x - mean) * rstd * w[c] + b[c]
    # sum over group = sum_c (rstd * w[c] * (sum_x_c - S*mean) + S*b[c])
    # We just do it directly.
    out_sum = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, group_size, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < group_size
        v = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        c_local = idx // S
        c_global = g * CPG + c_local
        w = tl.load(w_ptr + c_global, mask=mask, other=0.0)
        b = tl.load(b_ptr + c_global, mask=mask, other=0.0)
        y = (v - mean_v) * rstd * w + b
        y = tl.where(mask, y, 0.0)
        out_sum += y

    total = tl.sum(out_sum, axis=0)
    tl.store(partial_ptr + n * G + g, total)


@triton.jit
def finalize_mean_kernel(
    partial_ptr, out_ptr,
    N, G, inv_total,
    BLOCK_G: tl.constexpr,
):
    n = tl.program_id(0)
    idx = tl.arange(0, BLOCK_G)
    mask = idx < G
    v = tl.load(partial_ptr + n * G + idx, mask=mask, other=0.0)
    s = tl.sum(v, axis=0)
    tl.store(out_ptr + n, s * inv_total)


def _next_pow2(x):
    p = 1
    while p < x:
        p <<= 1
    return p


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.eps = 1e-5

    def forward(self, x):
        x = x.contiguous()
        N, IC, ID, IH, IW = x.shape
        KD = KH = KW = self.kernel_size
        OC = self.out_channels
        OD = ID - KD + 1
        OH = IH - KH + 1
        OW = IW - KW + 1

        y = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

        BLOCK_M = 32
        BLOCK_N = 128
        out_spatial = OD * OH * OW
        grid = (N, (OC + BLOCK_M - 1) // BLOCK_M, (out_spatial + BLOCK_N - 1) // BLOCK_N)

        conv3d_kernel[grid](
            x, self.conv.weight, self.conv.bias, y,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Fused GroupNorm + per-group sum
        G = self.num_groups
        CPG = OC // G
        S = OD * OH * OW

        partial = torch.empty((N, G), device=x.device, dtype=torch.float32)
        BLOCK = 1024
        gn_mean_kernel[(N * G,)](
            y, self.group_norm.weight, self.group_norm.bias, partial,
            N, G, CPG, S, self.eps,
            BLOCK=BLOCK, num_warps=4,
        )

        out = torch.empty(N, device=x.device, dtype=x.dtype)
        total = OC * S
        inv_total = 1.0 / total
        BLOCK_G = _next_pow2(G)
        finalize_mean_kernel[(N,)](
            partial, out, N, G, inv_total,
            BLOCK_G=BLOCK_G, num_warps=1,
        )
        return out