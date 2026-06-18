import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# ConvTranspose2d implemented as im2col-style GEMM with mean reduction fused.
# Output of conv_transpose: out[n, oc, oh, ow] = sum_{ic, kh, kw} x[n, ic, oh-kh, ow-kw] * w[ic, oc, kh, kw]
#   for stride=1, padding=0. OH = IH + KH - 1, OW = IW + KW - 1.
#
# We tile by (N, OC_tile, OHW_tile). Each program computes a [BLOCK_OC, BLOCK_HW] tile of output,
# accumulates the multiply-adds (full asymptotic work), and atomicAdds the partial sum (over the
# spatial tile) into a pooled buffer of shape (N, OC). This fuses the mean's spatial reduction
# into the convolution epilogue so the full (N, OC, OH, OW) tensor is never materialized.

@triton.jit
def conv_transpose2d_pool_kernel(
    x_ptr, w_ptr, cb_ptr, pooled_ptr,
    N, IC, IH, IW,
    OC, OH, OW, KH, KW,
    HAS_CB: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC

    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    OHW = OH * OW
    hw_mask = hw_offs < OHW

    oh = hw_offs // OW  # [BLOCK_HW]
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Loop over kh, kw, ic
    for kh in range(0, KH):
        ih = oh - kh  # [BLOCK_HW]
        ih_valid = (ih >= 0) & (ih < IH)
        for kw in range(0, KW):
            iw = ow - kw  # [BLOCK_HW]
            iw_valid = (iw >= 0) & (iw < IW)
            spatial_valid = ih_valid & iw_valid & hw_mask  # [BLOCK_HW]
            # Clamp indices for safe addressing
            ih_c = tl.where(ih_valid, ih, 0)
            iw_c = tl.where(iw_valid, iw, 0)
            for ic in range(0, IC):
                x_off = ((pid_n * IC + ic) * IH + ih_c) * IW + iw_c  # [BLOCK_HW]
                x_val = tl.load(x_ptr + x_off, mask=spatial_valid, other=0.0)
                w_off = ((ic * OC + oc_offs) * KH + kh) * KW + kw  # [BLOCK_OC]
                w_val = tl.load(w_ptr + w_off, mask=oc_mask, other=0.0)
                acc += w_val[:, None] * x_val[None, :]

    # Add conv bias if present
    if HAS_CB:
        cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc += cb[:, None]

    # zero-out invalid spatial elements then sum across BLOCK_HW
    full_mask = oc_mask[:, None] & hw_mask[None, :]
    acc = tl.where(full_mask, acc, 0.0)
    partial = tl.sum(acc, axis=1)  # [BLOCK_OC]

    # Atomic add to pooled (N, OC)
    pooled_off = pid_n * OC + oc_offs
    tl.atomic_add(pooled_ptr + pooled_off, partial, mask=oc_mask)


@triton.jit
def finalize_kernel(
    pooled_ptr, bias_ptr, out_ptr,
    N, OC, OHW,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    mask = oc_offs < OC
    p = tl.load(pooled_ptr + n * OC + oc_offs, mask=mask, other=0.0)
    p = p / OHW
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
        has_cb = cb is not None
        cb_c = cb.contiguous().cuda() if has_cb else x  # placeholder

        N, IC, IH, IW = x.shape
        IC2, OC, KH, KW = w.shape
        OH = IH + KH - 1
        OW = IW + KW - 1
        OHW = OH * OW

        pooled = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 32
        BLOCK_HW = 128

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OHW, BLOCK_HW))
        conv_transpose2d_pool_kernel[grid](
            x, w, cb_c, pooled,
            N, IC, IH, IW,
            OC, OH, OW, KH, KW,
            HAS_CB=has_cb,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            num_warps=4,
            num_stages=2,
        )

        bias_flat = self.bias.view(-1).contiguous().cuda()
        out = torch.empty((N,), device=x.device, dtype=torch.float32)

        BLOCK_OC_FIN = 1
        while BLOCK_OC_FIN < OC:
            BLOCK_OC_FIN *= 2

        finalize_kernel[(N,)](
            pooled, bias_flat, out,
            N, OC, OHW,
            BLOCK_OC=BLOCK_OC_FIN,
            num_warps=4,
        )

        return out.view(N, 1)