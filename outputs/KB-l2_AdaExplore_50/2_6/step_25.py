import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1, num_stages=2),
        triton.Config({}, num_warps=1, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=2),
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=4, num_stages=3),
    ],
    key=['N', 'OD', 'OH', 'OW', 'OC', 'IC'],
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
    K_VOL: tl.constexpr,  # IC * KD * KH * KW
):
    # grid: (OW, OD * OH, N)
    ow = tl.program_id(0)
    tmp = tl.program_id(1)
    oh = tmp % OH
    od = tmp // OH
    n = tl.program_id(2)

    # pre-pool position start
    d_start = od * P
    h_start = oh * P
    w_start = ow * P

    oc_offs = tl.arange(0, OC)
    k_offs = tl.arange(0, K_VOL)

    # bias load (constant per kernel)
    bias = tl.load(b_ptr + oc_offs)  # [OC]

    # weight load [OC, K_VOL] - hoisted outside pool loop
    w_off = oc_offs[:, None] * K_VOL + k_offs[None, :]
    w_vals = tl.load(w_ptr + w_off)  # [OC, K_VOL]

    # decode k_offs into (ic, kd, kh, kw) - constexpr math
    ic_idx = k_offs // (KD * KH * KW)
    rem = k_offs % (KD * KH * KW)
    kd_idx = rem // (KH * KW)
    rem2 = rem % (KH * KW)
    kh_idx = rem2 // KW
    kw_idx = rem2 % KW

    # accumulator for max over P*P*P window per channel
    max_vals = tl.full([OC], -float('inf'), dtype=tl.float32)

    # iterate over pooling window (P^3 conv outputs)
    for di in tl.static_range(P):
        for hi in tl.static_range(P):
            for wi in tl.static_range(P):
                d_conv = d_start + di
                h_conv = h_start + hi
                w_conv = w_start + wi

                # gather input [K_VOL]
                d_in = d_conv + kd_idx
                h_in = h_conv + kh_idx
                w_in = w_conv + kw_idx
                x_off = (n * IC * D_in * H_in * W_in
                         + ic_idx * (D_in * H_in * W_in)
                         + d_in * (H_in * W_in)
                         + h_in * W_in
                         + w_in)
                x_vals = tl.load(x_ptr + x_off)  # [K_VOL]

                # acc = bias + w_vals @ x_vals
                acc = bias + tl.sum(w_vals * x_vals[None, :], axis=1)

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

    K_VOL = IC * KD * KH * KW
    grid = (OW, OD * OH, N)
    fused_conv_softmax_pool_kernel[grid](
        x, weight, bias, out,
        N, IC,
        D_in, H_in, W_in,
        D_conv, H_conv, W_conv,
        OD, OH, OW,
        OC,
        KD, KH, KW,
        pool_factor,
        K_VOL,
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