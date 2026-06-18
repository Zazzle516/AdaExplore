import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


# Strategy: build on kernel #2 (best-performing in pool) but improve tile sizes
# and exploit the structure of the problem.
#
# Problem: ConvTranspose2d (stride=1, pad=0, k=3) -> mean over HW -> +bias -> LSE -> *10
# We must do the full conv-transpose multiply-adds (no algebraic shortcut).
# Fusion: never write the OH*OW conv output. Each program computes its tile of
# conv-transpose output, sums it spatially in registers, atomic-adds into (N, OC) pooled.
#
# Improvements over kernel #2:
#  - Larger BLOCK_HW (256) with larger BLOCK_OC (64) leveraging tl.dot tensor cores.
#  - Use BLOCK_IC = IC (64) so no inner ic loop, single dot per (kh,kw).
#  - Reduce atomic contention by larger tiles.

@triton.jit
def conv_transpose_mean_fused_kernel(
    x_ptr, w_ptr, cb_ptr, pooled_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    inv_HW,
    KH: tl.constexpr, KW: tl.constexpr,
    HAS_CONV_BIAS: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_oc = tl.program_id(1)
    pid_hw = tl.program_id(2)

    oc_offs = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    oc_mask = oc_offs < OC
    HW = OH * OW
    hw_mask = hw_offs < HW

    oh = hw_offs // OW
    ow = hw_offs % OW

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for kh in tl.static_range(0, KH):
        ih = oh - kh
        ih_valid = (ih >= 0) & (ih < IH)
        for kw in tl.static_range(0, KW):
            iw = ow - kw
            iw_valid = (iw >= 0) & (iw < IW)
            spatial_valid = ih_valid & iw_valid & hw_mask
            ih_c = tl.where(ih_valid, ih, 0)
            iw_c = tl.where(iw_valid, iw, 0)

            for ic_base in range(0, IC, BLOCK_IC):
                ic_offs = ic_base + tl.arange(0, BLOCK_IC)
                ic_mask = ic_offs < IC

                # x[n, ic, ih, iw]: [BLOCK_IC, BLOCK_HW]
                x_off = ((pid_n * IC + ic_offs[:, None]) * IH + ih_c[None, :]) * IW + iw_c[None, :]
                x_m = ic_mask[:, None] & spatial_valid[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_m, other=0.0)

                # w[ic, oc, kh, kw]: [BLOCK_IC, BLOCK_OC]
                w_off = ((ic_offs[:, None] * OC + oc_offs[None, :]) * KH + kh) * KW + kw
                w_m = ic_mask[:, None] & oc_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_m, other=0.0)

                acc += tl.dot(tl.trans(w_val), x_val)

    if HAS_CONV_BIAS:
        cb = tl.load(cb_ptr + oc_offs, mask=oc_mask, other=0.0)
        acc += cb[:, None]

    valid = oc_mask[:, None] & hw_mask[None, :]
    acc = tl.where(valid, acc, 0.0)

    partial_sum = tl.sum(acc, axis=1) * inv_HW

    pooled_off = pid_n * OC + oc_offs
    tl.atomic_add(pooled_ptr + pooled_off, partial_sum, mask=oc_mask)


@triton.jit
def add_bias_logsumexp_kernel(
    pooled_ptr, bias_ptr, out_ptr,
    N, OC,
    BLOCK_OC: tl.constexpr,
):
    n = tl.program_id(0)
    oc_offs = tl.arange(0, BLOCK_OC)
    mask = oc_offs < OC
    p = tl.load(pooled_ptr + n * OC + oc_offs, mask=mask, other=-float('inf'))
    b = tl.load(bias_ptr + oc_offs, mask=mask, other=0.0)
    v = tl.where(mask, p + b, -float('inf'))
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
        w = self.conv_transpose.weight.contiguous().cuda()  # [IC, OC, KH, KW]
        cb = self.conv_transpose.bias
        cb_c = cb.contiguous().cuda() if cb is not None else None

        N, IC, IH, IW = x.shape
        IC2, OC, KH, KW = w.shape
        OH = IH + KH - 1
        OW = IW + KW - 1

        pooled = torch.zeros((N, OC), device=x.device, dtype=torch.float32)

        BLOCK_OC = 64
        BLOCK_HW = 256
        BLOCK_IC = 64 if IC <= 64 else 32

        inv_HW = 1.0 / float(OH * OW)

        grid = (N, triton.cdiv(OC, BLOCK_OC), triton.cdiv(OH * OW, BLOCK_HW))
        conv_transpose_mean_fused_kernel[grid](
            x, w, cb_c if cb_c is not None else x, pooled,
            N, IC, IH, IW, OC, OH, OW,
            inv_HW,
            KH=KH, KW=KW,
            HAS_CONV_BIAS=(cb_c is not None),
            BLOCK_OC=BLOCK_OC, BLOCK_HW=BLOCK_HW, BLOCK_IC=BLOCK_IC,
            num_warps=8, num_stages=3,
        )

        bias_flat = self.bias.view(-1).contiguous().cuda()
        out = torch.empty((N,), device=x.device, dtype=torch.float32)
        BLOCK_OC2 = 1
        while BLOCK_OC2 < OC:
            BLOCK_OC2 *= 2
        add_bias_logsumexp_kernel[(N,)](
            pooled, bias_flat, out, N, OC, BLOCK_OC=BLOCK_OC2, num_warps=4
        )

        return out.view(N, 1)