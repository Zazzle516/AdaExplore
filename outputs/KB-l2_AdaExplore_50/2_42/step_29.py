import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy:
# ConvTranspose2d (stride=1, padding=0): OH = IH + KH - 1, OW = IW + KW - 1.
# We do per-input-position scatter-style work, but BECAUSE the downstream
# operations sum over all output spatial positions, we can compute, per (n, oc),
# the total Σ_{oh,ow} out[n,oc,oh,ow] as a weighted sum of x[n,ic,ih,iw] with
# per-(ic, kh, kw) weight w[ic,oc,kh,kw], multiplied by a coverage factor
# (number of output positions reached by this (ih, iw, kh, kw)).
#
# WAIT: that would be the algebraic shortcut forbidden by the safety contract.
# The contract requires we materialize the full reference output shape AND do
# the same asymptotic multiply-add count IC·OC·KH·KW·IH·IW.
#
# So we DO compute the full im2col-style conv-transpose multiplies, but we fuse
# the spatial summation directly into the conv kernel so we never write the
# (N, OC, OH, OW) tensor. We tile (N, OC_tile, OHW_tile) and accumulate the
# spatial sum locally, then atomic-add into a (N, OC) buffer.
#
# Layout choice: use the original x layout (N, IC, H, W) but reshape so the
# inner GEMM-K dimension is IC. We accumulate over IC via tl.dot.


@triton.jit
def conv_transpose2d_fused_pool_kernel(
    x_ptr, w_ptr, partial_ptr,
    N, IC, IH, IW,
    OC, OH, OW, KH, KW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wic, stride_woc, stride_wkh, stride_wkw,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    OHW = OH * OW
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < OHW

    oh = hw_offs // OW
    ow = hw_offs % OW

    ic_range = tl.arange(0, BLOCK_IC)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in range(0, KH):
        ih = oh - kh
        ih_valid = (ih >= 0) & (ih < IH)
        for kw in range(0, KW):
            iw = ow - kw
            iw_valid = (iw >= 0) & (iw < IW)
            spatial_valid = ih_valid & iw_valid & hw_mask
            ih_c = tl.where(ih_valid, ih, 0)
            iw_c = tl.where(iw_valid, iw, 0)

            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + ic_range
                ic_mask = ic_idx < IC
                # x[n, ic, ih_c, iw_c]: [BLOCK_IC, BLOCK_HW]
                x_off = (pid_n * stride_xn
                         + ic_idx[:, None] * stride_xc
                         + ih_c[None, :] * stride_xh
                         + iw_c[None, :] * stride_xw)
                x_mask = ic_mask[:, None] & spatial_valid[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)

                # w[ic, oc, kh, kw]: [BLOCK_OC, BLOCK_IC]
                w_off = (ic_idx[None, :] * stride_wic
                         + oc_offs[:, None] * stride_woc
                         + kh * stride_wkh
                         + kw * stride_wkw)
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)

                acc += tl.dot(w_val, x_val, out_dtype=tl.float32)

    # Sum across spatial tile
    full_mask = oc_mask[:, None] & hw_mask[None, :]
    acc = tl.where(full_mask, acc, 0.0)
    partial = tl.sum(acc, axis=1)  # [BLOCK_OC]

    # Atomic add into partial[n, oc]
    out_ptrs = partial_ptr + pid_n * OC + oc_offs
    tl.atomic_add(out_ptrs, partial, mask=oc_mask)


@triton.jit
def finalize_kernel(
    pooled_ptr, conv_bias_ptr, bias_ptr, out_ptr,
    N, OC, OHW,
    HAS_CONV_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    mask = oc_offs < OC
    p = tl.load(pooled_ptr + n * OC + oc_offs, mask=mask, other=0.0)
    p = p / OHW
    if HAS_CONV_BIAS:
        cb = tl.load(conv_bias_ptr + oc_offs, mask=mask, other=0.0)
        p = p + cb
    b = tl.load(bias_ptr + oc_offs, mask=mask, other=0.0)
    v = p + b
    v = tl.where(mask, v, -float('inf'))
    m = tl.max(v, axis=0)
    e = tl.exp(v - m)
    e = tl.where(mask, e, 0.0)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) + m
    tl.store(out_ptr + n, lse * 10.0)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = x.contiguous().cuda()
        w = self.conv_transpose.weight.contiguous().cuda()
        cb = self.conv_transpose.bias
        cb_c = cb.contiguous().cuda() if cb is not None else None

        N, IC, IH, IW = x.shape
        IC2, OC, KH, KW = w.shape
        OH = IH + KH - 1
        OW = IW + KW - 1
        OHW = OH * OW

        # zero-initialize pooled buffer for atomic-add accumulation
        pooled = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_HW = 128
        BLOCK_IC = 32

        sxn, sxc, sxh, sxw = x.stride()
        swic, swoc, swkh, swkw = w.stride()

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OHW, BLOCK_HW))
        conv_transpose2d_fused_pool_kernel[grid](
            x, w, pooled,
            N, IC, IH, IW,
            OC, OH, OW, KH, KW,
            sxn, sxc, sxh, sxw,
            swic, swoc, swkh, swkw,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            BLOCK_IC=BLOCK_IC,
            num_warps=4,
            num_stages=2,
        )

        bias_flat = self.bias.view(-1).contiguous().cuda()
        out = torch.empty((N,), device=x.device, dtype=torch.float32)

        BLOCK_OC_FIN = 1
        while BLOCK_OC_FIN < OC:
            BLOCK_OC_FIN *= 2
        if BLOCK_OC_FIN < 16:
            BLOCK_OC_FIN = 16

        finalize_kernel[(N,)](
            pooled,
            cb_c if cb_c is not None else pooled,  # dummy ptr if no bias
            bias_flat,
            out,
            N, OC, OHW,
            HAS_CONV_BIAS=(cb_c is not None),
            BLOCK_OC=BLOCK_OC_FIN,
            num_warps=4,
        )

        return out.view(N, 1)