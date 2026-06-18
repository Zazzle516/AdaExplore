import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ============================================================
# Conv3d forward as implicit GEMM
#   - One program per (N, OC tile, output-spatial tile)
#   - Implicit im2col: compute input offsets in-kernel
#   - Bias added in epilogue
#   - Also writes per-(N, OC) partial sum (sum over D*H*W) to use
#     for the final mean reduction.
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2, num_stages=2),
    ],
    key=['N', 'OC', 'IC', 'D_out', 'H_out', 'W_out'],
)
@triton.jit
def conv3d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr, partial_sum_ptr,
    N, IC, D, H, W,
    OC,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    D_out, H_out, W_out,
    S_out,  # D_out*H_out*W_out
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wo, stride_wi, stride_wd, stride_wh, stride_ww,
    stride_on, stride_oc, stride_od, stride_oh, stride_ow,
    BLOCK_M: tl.constexpr,  # output spatial tile
    BLOCK_N: tl.constexpr,  # OC tile
):
    pid_n = tl.program_id(0)            # batch
    pid_oc = tl.program_id(1)           # OC tile
    pid_sp = tl.program_id(2)           # spatial tile

    offs_m = pid_sp * BLOCK_M + tl.arange(0, BLOCK_M)        # spatial idx
    offs_n = pid_oc * BLOCK_N + tl.arange(0, BLOCK_N)        # OC idx
    mask_m = offs_m < S_out
    mask_n = offs_n < OC

    # decompose spatial idx into (od, oh, ow)
    HW_out = H_out * W_out
    od = offs_m // HW_out
    rem = offs_m % HW_out
    oh = rem // W_out
    ow = rem % W_out

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over IC, KD, KH, KW
    for ic in range(0, IC):
        for kd in range(0, KD):
            for kh in range(0, KH):
                for kw in range(0, KW):
                    id_ = od + kd
                    ih_ = oh + kh
                    iw_ = ow + kw
                    # load input: shape [BLOCK_M]
                    x_off = (pid_n * stride_xn
                             + ic * stride_xc
                             + id_ * stride_xd
                             + ih_ * stride_xh
                             + iw_ * stride_xw)
                    x_vals = tl.load(x_ptr + x_off, mask=mask_m, other=0.0)
                    # load weight: shape [BLOCK_N]
                    w_off = (offs_n * stride_wo
                             + ic * stride_wi
                             + kd * stride_wd
                             + kh * stride_wh
                             + kw * stride_ww)
                    w_vals = tl.load(w_ptr + w_off, mask=mask_n, other=0.0)
                    # outer product accumulate
                    acc += x_vals[:, None] * w_vals[None, :]

    # add bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_vals[None, :]

    # write output [BLOCK_M, BLOCK_N] -> out[N, OC, S_out]
    out_off = (pid_n * stride_on
               + offs_n[None, :] * stride_oc
               + offs_m[:, None])  # contiguous spatial
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_off, acc, mask=out_mask)

    # partial sum over spatial tile per OC -> atomic add to partial_sum[N, OC]
    acc_masked = tl.where(out_mask, acc, 0.0)
    psum = tl.sum(acc_masked, axis=0)  # [BLOCK_N]
    ps_off = pid_n * OC + offs_n
    tl.atomic_add(partial_sum_ptr + ps_off, psum, mask=mask_n)


# ============================================================
# GroupNorm stats: compute mean & rstd per (N, G), where each
# group has channels_per_group * S_out elements
# ============================================================
@triton.jit
def groupnorm_stats_kernel(
    x_ptr,          # conv output [N, C, S]
    mean_ptr,       # [N, G]
    rstd_ptr,       # [N, G]
    N, C, S,
    G: tl.constexpr,
    CPG: tl.constexpr,   # channels per group = C / G
    eps,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // G
    g = pid % G

    group_size = CPG * S
    base = n * C * S + g * CPG * S

    sum_ = tl.zeros((), dtype=tl.float32)
    sumsq_ = tl.zeros((), dtype=tl.float32)

    num_iters = (group_size + BLOCK_S - 1) // BLOCK_S
    for i in range(0, num_iters):
        offs = i * BLOCK_S + tl.arange(0, BLOCK_S)
        mask = offs < group_size
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        sum_ += tl.sum(vals, axis=0)
        sumsq_ += tl.sum(vals * vals, axis=0)

    mean = sum_ / group_size
    var = sumsq_ / group_size - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(mean_ptr + pid, mean)
    tl.store(rstd_ptr + pid, rstd)


# ============================================================
# Final fused kernel:
#   - Apply group-norm scaling to conv output (read only, no store)
#   - Compute mean over (C, D, H, W) per batch -> output [N]
#
# Uses precomputed per-(N, OC) partial sums of conv output:
#     conv_psum[n, c] = sum_{spatial} conv_out[n, c, :]
# After GN:  y[n,c,s] = (conv_out[n,c,s] - mean[n,g(c)]) * rstd[n,g(c)] * gamma[c] + beta[c]
# Sum_{c,s} y[n,c,s] = sum_c [ rstd[n,g(c)]*gamma[c] * (conv_psum[n,c] - mean[n,g(c)] * S)
#                              + beta[c] * S ]
# That's a tiny (N, C) -> (N,) reduce. No need to re-read conv output!
# ============================================================
@triton.jit
def final_mean_kernel(
    conv_psum_ptr,   # [N, C]
    mean_ptr,        # [N, G]
    rstd_ptr,        # [N, G]
    gamma_ptr,       # [C]
    beta_ptr,        # [C]
    out_ptr,         # [N]
    N, C,
    G: tl.constexpr,
    CPG: tl.constexpr,
    S,               # D_out*H_out*W_out
    total_elems,     # C * S
    BLOCK_C: tl.constexpr,
):
    n = tl.program_id(0)

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # group index for each c
    g_idx = offs_c // CPG

    psum = tl.load(conv_psum_ptr + n * C + offs_c, mask=mask_c, other=0.0)
    gamma = tl.load(gamma_ptr + offs_c, mask=mask_c, other=0.0)
    beta = tl.load(beta_ptr + offs_c, mask=mask_c, other=0.0)

    mean = tl.load(mean_ptr + n * G + g_idx, mask=mask_c, other=0.0)
    rstd = tl.load(rstd_ptr + n * G + g_idx, mask=mask_c, other=0.0)

    # per-c contribution to total sum
    contrib = rstd * gamma * (psum - mean * S) + beta * S
    contrib = tl.where(mask_c, contrib, 0.0)
    total = tl.sum(contrib, axis=0)

    out_val = total / total_elems
    tl.store(out_ptr + n, out_val)


def conv3d_triton(x, weight, bias):
    # x: [N, IC, D, H, W], weight: [OC, IC, KD, KH, KW]
    N, IC, D, H, W = x.shape
    OC, _, KD, KH, KW = weight.shape
    D_out = D - KD + 1
    H_out = H - KH + 1
    W_out = W - KW + 1
    S_out = D_out * H_out * W_out

    x = x.contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()

    out = torch.empty((N, OC, S_out), device=x.device, dtype=x.dtype)
    partial_sum = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

    grid = lambda meta: (N, triton.cdiv(OC, meta['BLOCK_N']), triton.cdiv(S_out, meta['BLOCK_M']))
    conv3d_kernel[grid](
        x, weight, bias, out, partial_sum,
        N, IC, D, H, W,
        OC,
        KD, KH, KW,
        D_out, H_out, W_out,
        S_out,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3), weight.stride(4),
        out.stride(0), out.stride(1), 1, 1, 1,  # last 3 unused; spatial is contiguous
    )

    return out, partial_sum, (D_out, H_out, W_out)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_groups):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(num_groups, out_channels)
        self.num_groups = num_groups
        self.out_channels = out_channels

    def forward(self, x):
        x = x.contiguous()
        N = x.shape[0]
        C = self.out_channels
        G = self.num_groups
        CPG = C // G

        # 1) Conv3d via Triton; also produces per-(N, OC) partial sums
        conv_out_flat, conv_psum, (D_out, H_out, W_out) = conv3d_triton(
            x, self.conv.weight, self.conv.bias
        )
        S = D_out * H_out * W_out

        # 2) GroupNorm stats per (N, G)
        mean = torch.empty((N, G), device=x.device, dtype=torch.float32)
        rstd = torch.empty((N, G), device=x.device, dtype=torch.float32)
        # pick BLOCK_S
        group_size = CPG * S
        BLOCK_S = 1024
        groupnorm_stats_kernel[(N * G,)](
            conv_out_flat, mean, rstd,
            N, C, S,
            G, CPG,
            float(self.group_norm.eps),
            BLOCK_S=BLOCK_S,
            num_warps=8,
        )

        # 3) Final fused mean using analytic sum
        out = torch.empty((N,), device=x.device, dtype=x.dtype)
        # next pow2 >= C
        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2
        total_elems = C * S
        final_mean_kernel[(N,)](
            conv_psum, mean, rstd,
            self.group_norm.weight, self.group_norm.bias,
            out,
            N, C,
            G, CPG,
            S, total_elems,
            BLOCK_C=BLOCK_C,
            num_warps=2,
        )
        return out