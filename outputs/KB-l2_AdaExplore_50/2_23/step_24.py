import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# -----------------------------------------------------------------------------
# Conv3d kernel: implicit im2col GEMM, tiled along the contiguous OW axis
# Output layout: (N, OC, OD, OH, OW) contiguous
# Grid: (N, ceil(OC/BLOCK_M), OD*OH)
# -----------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_W': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_W': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_W': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_W': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_W': 64}, num_warps=8, num_stages=2),
    ],
    key=['OC', 'OD', 'OH', 'OW', 'IC'],
)
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr,
    N, IC: tl.constexpr, ID, IH, IW,
    OC, OD, OH, OW,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_r = tl.program_id(2)   # row index over OD*OH

    od = pid_r // OH
    oh = pid_r - od * OH

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_w = tl.arange(0, BLOCK_W)
    mask_m = offs_m < OC
    mask_w = offs_w < OW

    K: tl.constexpr = IC * KD * KH * KW

    # Preload weight tile (BLOCK_M, K) once.
    k_idx = tl.arange(0, K)
    w_addr = offs_m[:, None] * K + k_idx[None, :]
    w_tile = tl.load(w_ptr + w_addr, mask=mask_m[:, None], other=0.0)  # (BLOCK_M, K)

    acc = tl.zeros((BLOCK_M, BLOCK_W), dtype=tl.float32)

    x_n_base = pid_n * (IC * ID * IH * IW)

    # K index decomposition: k = ((ic*KD + kd)*KH + kh)*KW + kw
    # Iterate over (ic, kd, kh, kw) and gather along ow
    for ic in tl.static_range(0, IC):
        for kd in tl.static_range(0, KD):
            for kh in tl.static_range(0, KH):
                for kw in tl.static_range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = offs_w + kw
                    x_addr = (x_n_base
                              + ic * (ID * IH * IW)
                              + id_ * (IH * IW)
                              + ih_ * IW
                              + iw_)
                    x_val = tl.load(x_ptr + x_addr, mask=mask_w, other=0.0)  # (BLOCK_W,)
                    k = ((ic * KD + kd) * KH + kh) * KW + kw
                    w_col = w_tile[:, k]  # (BLOCK_M,)
                    acc += w_col[:, None] * x_val[None, :]

    bias = tl.load(b_ptr + offs_m, mask=mask_m, other=0.0)
    acc = acc + bias[:, None]

    out_spatial = OD * OH * OW
    out_idx = (pid_n * OC * out_spatial
               + offs_m[:, None] * out_spatial
               + (od * OH + oh) * OW
               + offs_w[None, :])
    tl.store(y_ptr + out_idx, acc, mask=mask_m[:, None] & mask_w[None, :])


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

        grid = lambda META: (N, (OC + META['BLOCK_M'] - 1) // META['BLOCK_M'], OD * OH)
        conv3d_kernel[grid](
            x, self.conv.weight, self.conv.bias, y,
            N, IC, ID, IH, IW,
            OC, OD, OH, OW,
            KD, KH, KW,
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