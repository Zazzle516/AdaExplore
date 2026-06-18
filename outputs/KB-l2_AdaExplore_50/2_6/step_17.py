import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=4),
        triton.Config({}, num_warps=8, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
        triton.Config({}, num_warps=8, num_stages=4),
    ],
    key=['N', 'OD', 'OH', 'OW'],
)
@triton.jit
def fused_conv_softmax_pool_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, IC: tl.constexpr,
    D_in, H_in, W_in,
    D_conv, H_conv, W_conv,
    OD, OH, OW,
    OC: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    P: tl.constexpr,
):
    # one program per (n, od, oh, ow); computes OC outputs
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    # pre-pool position start
    d_start = od * P
    h_start = oh * P
    w_start = ow * P

    oc_offs = tl.arange(0, OC)

    # accumulator for max over P*P*P window per channel
    max_vals = tl.full([OC], -float('inf'), dtype=tl.float32)

    # bias load (constant per kernel)
    bias = tl.load(b_ptr + oc_offs)  # [OC]

    # Hoist: load full weight tile [OC, IC*KD*KH*KW] once
    K_VOL: tl.constexpr = IC * KD * KH * KW
    k_offs = tl.arange(0, K_VOL)
    w_full_off = oc_offs[:, None] * K_VOL + k_offs[None, :]
    w_full = tl.load(w_ptr + w_full_off)  # [OC, K_VOL]

    # Precompute k-axis deltas (constant w.r.t. di/hi/wi)
    DHW = D_in * H_in * W_in
    HW = H_in * W_in
    ic_idx = k_offs // (KD * KH * KW)
    rem = k_offs % (KD * KH * KW)
    kd_idx = rem // (KH * KW)
    rem2 = rem % (KH * KW)
    kh_idx = rem2 // KW
    kw_idx = rem2 % KW
    k_delta = ic_idx * DHW + kd_idx * HW + kh_idx * W_in + kw_idx  # [K_VOL]

    n_base = n * IC * DHW + d_start * HW + h_start * W_in + w_start

    # iterate over pooling window (P^3 conv outputs)
    for di in tl.static_range(P):
        for hi in tl.static_range(P):
            for wi in tl.static_range(P):
                base_iter = n_base + di * HW + hi * W_in + wi
                x_off = base_iter + k_delta
                x_patch = tl.load(x_ptr + x_off)  # [K_VOL]

                # acc[oc] = bias + sum_k w_full[oc,k] * x_patch[k]
                acc = bias + tl.sum(w_full * x_patch[None, :], axis=1)

                # softmax across OC
                m = tl.max(acc, axis=0)
                ex = tl.exp(acc - m)
                s = tl.sum(ex, axis=0)
                sm = ex / s

                max_vals = tl.maximum(max_vals, sm)

    # store
    out_base = (n * OC * OD * OH * OW
                + od * OH * OW
                + oh * OW
                + ow)
    out_offs = out_base + oc_offs * (OD * OH * OW)
    tl.store(out_ptr + out_offs, max_vals)


def fused_conv_softmax_pool(x, weight, bias, pool_factor):
    N, IC, D_in, H_in, W_in = x.shape
    OC, _, KD, KH, KW = weight.shape

    D_conv = D_in - KD + 1
    H_conv = H_in - KH + 1
    W_conv = W_in - KW + 1

    OD = D_conv // pool_factor
    OH = H_conv // pool_factor
    OW = W_conv // pool_factor

    out = torch.empty((N, OC, OD, OH, OW), device=x.device, dtype=x.dtype)

    grid = (N * OD * OH * OW,)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC,
        D_in, H_in, W_in,
        D_conv, H_conv, W_conv,
        OD, OH, OW,
        OC,
        KD, KH, KW,
        pool_factor,
    )
    return out


@triton.jit
def softmax_pool_kernel(
    x_ptr, out_ptr,
    N, C, D, H, W,
    OD, OH, OW,
    P: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    ow = pid % OW
    tmp = pid // OW
    oh = tmp % OH
    tmp = tmp // OH
    od = tmp % OD
    n = tmp // OD

    d_start = od * P
    h_start = oh * P
    w_start = ow * P

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    max_vals = tl.full([BLOCK_C], -float('inf'), dtype=tl.float32)

    for di in tl.static_range(P):
        for hi in tl.static_range(P):
            for wi in tl.static_range(P):
                d = d_start + di
                h = h_start + hi
                w = w_start + wi
                base = n * C * D * H * W + d * H * W + h * W + w
                x_offs = base + c_offs * (D * H * W)
                vals = tl.load(x_ptr + x_offs, mask=c_mask, other=-float('inf'))
                m = tl.max(tl.where(c_mask, vals, -float('inf')), axis=0)
                ex = tl.exp(vals - m)
                ex = tl.where(c_mask, ex, 0.0)
                s = tl.sum(ex, axis=0)
                sm = ex / s
                max_vals = tl.maximum(max_vals, sm)

    out_base = n * C * OD * OH * OW + od * OH * OW + oh * OW + ow
    out_offs = out_base + c_offs * (OD * OH * OW)
    tl.store(out_ptr + out_offs, max_vals, mask=c_mask)


def softmax_pool_fused(x, pool_factor):
    N, C, D, H, W = x.shape
    OD = D // pool_factor
    OH = H // pool_factor
    OW = W // pool_factor
    out = torch.empty((N, C, OD, OH, OW), device=x.device, dtype=x.dtype)
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2
    BLOCK_C = max(BLOCK_C, 16)
    grid = (N * OD * OH * OW,)
    softmax_pool_kernel[grid](
        x, out,
        N, C, D, H, W,
        OD, OH, OW,
        P=pool_factor,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool_kernel_size = pool_kernel_size
        self.pool_factor = pool_kernel_size * pool_kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous()
        weight = self.conv.weight.contiguous()
        bias = self.conv.bias.contiguous()

        N, IC, D_in, H_in, W_in = x.shape
        KD = KH = KW = self.kernel_size
        D_conv = D_in - KD + 1
        H_conv = H_in - KH + 1
        W_conv = W_in - KW + 1
        pf = self.pool_factor

        if (D_conv % pf == 0 and H_conv % pf == 0 and W_conv % pf == 0):
            try:
                return fused_conv_softmax_pool(x, weight, bias, pf)
            except Exception:
                pass

        x = self.conv(x)
        N2, C2, D2, H2, W2 = x.shape
        if D2 % pf == 0 and H2 % pf == 0 and W2 % pf == 0:
            x = x.contiguous()
            return softmax_pool_fused(x, pf)
        x = torch.softmax(x, dim=1)
        x = F.max_pool3d(x, self.pool_kernel_size)
        x = F.max_pool3d(x, self.pool_kernel_size)
        return x