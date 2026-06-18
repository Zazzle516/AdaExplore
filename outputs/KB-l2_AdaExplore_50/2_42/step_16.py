import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Fused ConvTranspose2d + spatial mean.
#
# For ConvTranspose2d with stride=1, padding=0:
#   y[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh - kh, ow - kw] * w[ic, oc, kh, kw] + b[oc]
# (with valid bounds on (oh-kh, ow-kw))
#
# H_out = H_in + KH - 1, W_out = W_in + KW - 1.
#
# mean_{oh,ow}(y[n,oc]) = (1/(H_out*W_out)) * sum_{oh,ow} y[n,oc,oh,ow]
#
# sum_{oh,ow} y[n,oc,oh,ow]
#   = sum_{ic,kh,kw} w[ic,oc,kh,kw] * sum_{oh,ow valid} x[n,ic,oh-kh,ow-kw]
#                                      + (H_out*W_out) * b[oc]
#
# But for stride=1, padding=0: as kh,kw range over [0,KH)x[0,KW), every
# input pixel x[n,ic,h,w] contributes once to the sum for (kh,kw) (since
# oh = h+kh ranges over [kh, kh+H_in) which is within [0, H_out) ).
# So sum_{oh,ow valid} x[n,ic,oh-kh,ow-kw] = sum_{h,w} x[n,ic,h,w]  -- same for all (kh,kw)!
#
# That would let us factor:
#   sum_y[n,oc] = (sum_ic SX[n,ic] * W_sum[ic,oc]) + HoutWout * b[oc]
#                where W_sum[ic,oc] = sum_{kh,kw} w[ic,oc,kh,kw], SX[n,ic] = sum_{h,w} x[n,ic,h,w]
#
# However the safety contract FORBIDS pre-reducing along axes that
# downstream reductions collapse. So we must keep the multiply-add count
# at the full O(N * IC * OC * KH * KW * H_in * W_in).
#
# Legitimate fusion: each (n, oc) program performs the full
# sum_{ic, kh, kw, h, w} x[n,ic,h,w] * w[ic,oc,kh,kw] * count(kh,kw)
# where count(kh,kw) = number of valid (oh,ow) such that 0<=oh-kh<H_in,
# 0<=ow-kw<W_in AND 0<=oh<H_out, 0<=ow<W_out — for stride=1 pad=0 this
# is H_in * W_in for every (kh,kw). We still iterate over (kh, kw) and
# (h, w) and (ic) explicitly, performing the full multiply-add count.
#
# That is HEAVY. So we organize: tile (h, w) into blocks. One program per
# (n, oc_block). Inner loop: ic, then h-tile, multiply-add x*w over kh,kw.
#
# Actually the simplest faithful structure: one program per (n, oc).
# Loop over ic, kh, kw — for each, compute partial = sum_{h,w} x[n,ic,h,w]
# loaded fresh each iteration, multiplied by w[ic,oc,kh,kw]. The fact that
# this partial is the same across (kh,kw) is just a property of the math —
# we still do KH*KW separate multiply-adds on it. This keeps total
# multiply-add count = N * OC * IC * KH * KW + the H*W reductions.
#
# Wait — re-reading the contract: "Specifically forbidden ... precomputing
# sum_x[n,ic] = Σ x[n,ic,h,w], then running a smaller GEMM/matvec". So we
# can't pre-reduce x once and reuse. We must do the per-(kh,kw) reductions
# each as separate work.
#
# OK so honest implementation: one program per (n, oc_tile). Loop over
# (ic, kh, kw). Inside: stream over (h, w) tiles of x[n, ic], accumulate
# the sum, multiply by w[ic, oc_tile, kh, kw], add to per-oc accumulator.
# This does N*OC*IC*KH*KW*H*W multiply-adds — same as the reference.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 16384}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'IC', 'OC', 'KHW'],
)
@triton.jit
def _fused_convt_mean_kernel(
    x_ptr,           # [N, IC, H, W]
    w_ptr,           # [IC, OC, KH, KW]
    b_ptr,           # [OC]
    out_ptr,         # [N, OC]
    N, IC, OC,
    HW,              # H_in * W_in
    H_out, W_out,
    KH, KW, KHW,     # KH*KW
    stride_xn, stride_xc,
    stride_wic, stride_woc,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)

    HoutWout = H_out * W_out
    inv = 1.0 / HoutWout.to(tl.float32)

    # accumulator for sum over (h,w) of x[n,ic,h,w]  -- but we mustn't
    # reuse it. Instead, accumulate sum_y for this (n, oc):
    #   sum_y = sum_{ic} (sum_{kh,kw} w[ic,oc,kh,kw]) * Sx[n,ic]
    # That IS pre-reducing. To avoid: do the kh*kw multiply-adds separately
    # against the loaded x tile each time.
    #
    # Per ic: stream x tiles. For each tile, for each (kh,kw), accumulate
    # tile_sum_x * w[ic,oc,kh,kw]. We compute tile_sum_x once per tile
    # then multiply by KH*KW different weights — that's fine, it's just
    # the standard GEMM reuse pattern: tile_sum_x is a single value, then
    # KH*KW MACs. Total MACs = N*OC*IC*KH*KW*(num_tiles) ~ matches the
    # reference flop count for the conv itself? The reference conv does
    # N*OC*IC*KH*KW*H_out*W_out ~ N*OC*IC*KH*KW*HW MACs. Here we do
    # N*OC*IC*KH*KW MACs with the per-tile sum, so we're under by HW.
    #
    # That means this IS using the math identity — it's the forbidden
    # shortcut. We need to NOT collapse the spatial dim before applying
    # the kh,kw weights.
    #
    # Honest version: for each (kh, kw), separately stream x and form
    # sum_{h,w} x[n,ic,h,w] (this is the same value but loaded fresh).
    # That gives N*OC*IC*KH*KW*HW loads and MACs — matches reference.

    sum_y = tl.zeros((1,), dtype=tl.float32)

    for ic in range(0, IC):
        x_base = pid_n * stride_xn + ic * stride_xc
        for k in range(0, KHW):
            # Sum over h,w of x[n,ic,:,:]
            s = tl.zeros((BLOCK_HW,), dtype=tl.float32)
            off = 0
            while off < HW:
                idx = off + tl.arange(0, BLOCK_HW)
                mask = idx < HW
                v = tl.load(x_ptr + x_base + idx, mask=mask, other=0.0)
                s += v
                off += BLOCK_HW
            tile_sum = tl.sum(s, axis=0)
            # weight w[ic, pid_oc, kh, kw]
            w_off = ic * stride_wic + pid_oc * stride_woc + k
            wv = tl.load(w_ptr + w_off)
            sum_y += tile_sum * wv

    # bias contribution: bias adds to every output position
    bv = tl.load(b_ptr + pid_oc)
    sum_y += bv * HoutWout.to(tl.float32)

    mean = sum_y * inv
    tl.store(out_ptr + pid_n * OC + pid_oc, mean)


# Simple per-(n, oc) reduction of a precomputed conv_transpose output.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _mean_kernel(
    y_ptr, out_ptr, HW,
    stride_n, stride_c, OC,
    BLOCK: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    base = pid_n * stride_n + pid_c * stride_c
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    off = 0
    while off < HW:
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(y_ptr + base + idx, mask=mask, other=0.0)
        acc += v
        off += BLOCK
    total = tl.sum(acc, axis=0)
    mean = total / HW.to(tl.float32)
    tl.store(out_ptr + pid_n * OC + pid_c, mean)


@triton.jit
def _lse_kernel(
    m_ptr, bias_ptr, out_ptr, OC,
    BLOCK_OC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK_OC)
    mask = offs < OC
    m = tl.load(m_ptr + pid * OC + offs, mask=mask, other=-float('inf'))
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    z = m + b
    z_masked = tl.where(mask, z, -float('inf'))
    mx = tl.max(z_masked, axis=0)
    e = tl.exp(z_masked - mx)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    lse = mx + tl.log(s)
    tl.store(out_ptr + pid, lse * 10.0)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # Heavy op: actual conv_transpose. Materialize full output.
        y = self.conv_transpose(x)  # [N, OC, H_out, W_out]
        N, OC, H, W = y.shape
        HW = H * W
        y = y.contiguous()

        m = torch.empty((N, OC), device=y.device, dtype=torch.float32)
        _mean_kernel[(N, OC)](
            y, m, HW,
            y.stride(0), y.stride(1), OC,
        )

        out = torch.empty((N,), device=y.device, dtype=torch.float32)
        bias_flat = self.bias.view(-1).contiguous()

        BLOCK_OC = 1
        while BLOCK_OC < OC:
            BLOCK_OC *= 2

        _lse_kernel[(N,)](
            m, bias_flat, out, OC,
            BLOCK_OC=BLOCK_OC,
            num_warps=4,
        )

        return out.view(N, 1)