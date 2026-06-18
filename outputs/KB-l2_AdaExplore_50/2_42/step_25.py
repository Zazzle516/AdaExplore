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
    x_ptr, w_ptr, partial_ptr,
    N, IC, IH, IW,
    OC, OH, OW, KH, KW,
    NUM_HW: tl.constexpr,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
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

    ic_range = tl.arange(0, BLOCK_IC)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    # Loop over kh, kw, ic_blocks
    for kh in range(0, KH):
        ih = oh - kh  # [BLOCK_HW]
        ih_valid = (ih >= 0) & (ih < IH)
        for kw in range(0, KW):
            iw = ow - kw  # [BLOCK_HW]
            iw_valid = (iw >= 0) & (iw < IW)
            spatial_valid = ih_valid & iw_valid & hw_mask  # [BLOCK_HW]
            ih_c = tl.where(ih_valid, ih, 0)
            iw_c = tl.where(iw_valid, iw, 0)
            for ic_start in range(0, IC, BLOCK_IC):
                ic_idx = ic_start + ic_range  # [BLOCK_IC]
                ic_mask = ic_idx < IC
                # x_val: [BLOCK_IC, BLOCK_HW]
                x_off = ((pid_n * IC + ic_idx[:, None]) * IH + ih_c[None, :]) * IW + iw_c[None, :]
                x_mask = ic_mask[:, None] & spatial_valid[None, :]
                x_val = tl.load(x_ptr + x_off, mask=x_mask, other=0.0)
                # w_val: [BLOCK_OC, BLOCK_IC]
                # weight shape: [IC, OC, KH, KW]
                w_off = ((ic_idx[None, :] * OC + oc_offs[:, None]) * KH + kh) * KW + kw
                w_mask = oc_mask[:, None] & ic_mask[None, :]
                w_val = tl.load(w_ptr + w_off, mask=w_mask, other=0.0)
                acc += tl.dot(w_val, x_val, out_dtype=tl.float32)

    # zero-out invalid spatial elements then sum across BLOCK_HW
    full_mask = oc_mask[:, None] & hw_mask[None, :]
    acc = tl.where(full_mask, acc, 0.0)
    partial = tl.sum(acc, axis=1)  # [BLOCK_OC]

    # Write partial[n, oc, pid_hw]
    out_off = (pid_n * OC + oc_offs) * NUM_HW + pid_hw
    tl.store(partial_ptr + out_off, partial, mask=oc_mask)


@triton.jit
def reduce_partials_kernel(
    partial_ptr, pooled_ptr,
    N, OC, NUM_HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # n*OC + oc
    offs = tl.arange(0, BLOCK)
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for start in range(0, NUM_HW, BLOCK):
        idx = start + offs
        mask = idx < NUM_HW
        v = tl.load(partial_ptr + pid * NUM_HW + idx, mask=mask, other=0.0)
        acc += v
    s = tl.sum(acc, axis=0)
    tl.store(pooled_ptr + pid, s)


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

        N, IC, IH, IW = x.shape
        IC2, OC, KH, KW = w.shape
        OH = IH + KH - 1
        OW = IW + KW - 1
        OHW = OH * OW

        # Worst-case BLOCK_HW from autotune configs is 256, but we need to know NUM_HW
        # before launching. Since BLOCK_HW is chosen by autotune, we compute NUM_HW
        # inside the kernel launch grid lambda using meta.
        # However NUM_HW is a constexpr -> we precompute for each possible BLOCK_HW.
        # Simpler: pre-allocate partial assuming a fixed max, but NUM_HW is part of indexing.
        # Use a callback grid and pass NUM_HW via meta:
        # Approach: launch with a fixed BLOCK_HW selection via autotune; allocate partial
        # after autotune chooses. We can use triton's grid lambda with meta.

        # Use a holder for partial allocated lazily; do via a wrapping function.
        # Simplest: try each config's BLOCK_HW results in different NUM_HW. We allocate
        # max possible (NUM_HW for smallest BLOCK_HW). But constexpr NUM_HW must match.
        # Solution: precompute NUM_HW per BLOCK_HW and pass.
        # Instead, just disable autotune by picking one good config manually:

        BLOCK_OC = 64
        BLOCK_HW = 128
        BLOCK_IC = 32
        NUM_HW = triton.cdiv(OHW, BLOCK_HW)

        partial = torch.empty((N, OC, NUM_HW), device=x.device, dtype=torch.float32)

        grid = (N, triton.cdiv(OC, BLOCK_OC), NUM_HW)
        conv_transpose2d_pool_kernel[grid](
            x, w, partial,
            N, IC, IH, IW,
            OC, OH, OW, KH, KW,
            NUM_HW=NUM_HW,
            BLOCK_OC=BLOCK_OC,
            BLOCK_HW=BLOCK_HW,
            BLOCK_IC=BLOCK_IC,
        )

        # Reduce partials over NUM_HW into pooled[N,OC]
        pooled = torch.empty((N, OC), device=x.device, dtype=torch.float32)
        # Add conv bias here if present (much cheaper than inside main kernel)
        BLOCK_RED = 1
        while BLOCK_RED < min(NUM_HW, 1024):
            BLOCK_RED *= 2
        reduce_partials_kernel[(N * OC,)](
            partial, pooled,
            N, OC, NUM_HW,
            BLOCK=BLOCK_RED,
            num_warps=4,
        )
        if cb is not None:
            pooled = pooled + cb.view(1, -1).to(pooled.device)

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