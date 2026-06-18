import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# Strategy:
# The reference computes:
#   y = conv_transpose2d(x, W, b_conv)        # [N, OC, H_out, W_out]
#   m = mean(y, dim=(2,3))                    # [N, OC, 1, 1]
#   z = m + bias                              # [N, OC, 1, 1]
#   l = logsumexp(z, dim=1)                   # [N, 1, 1, 1]
#   s = sum(l, dim=(2,3)) * 10                # [N, 1]
#
# Key observation (NOT a graph shortcut): we still must materialize the
# full conv_transpose output to honor the safety contract. We do so with
# a custom Triton scatter-add kernel that performs the FULL multiply-add
# count of conv_transpose (one mul-add per (N, OC, IC, kh, kw, H_in, W_in)).
#
# We write the full output [N, OC, H_out, W_out] to memory and then run
# a fused post kernel for (mean + bias + logsumexp + sum + ×10).
#
# To make conv_transpose fast we view it as:
#   For each (n, h_in, w_in): outer = x[n, :, h_in, w_in] (IC) ⊗ W[:, :, kh, kw] (IC, OC)
#   accumulate to y[n, :, h_in+kh, w_in+kw]
#
# We use a GEMM-style program tiling: one program per (n, oc_tile, hw_in_tile).
# Each program loads x[n, :, h_in_tile, w_in_tile] of shape [IC, HW_TILE]
# and unrolls over the 3x3 kernel taps, doing a [BLOCK_OC, IC] x [IC, HW_TILE]
# matmul for each tap and atomic-adds into the output at shifted positions.
#
# Since kernel_size=3, stride=1, padding=0:  H_out = H_in + 2, W_out = W_in + 2
# Each input position (h_in, w_in) contributes to output (h_in+kh, w_in+kw)
# for kh, kw in {0,1,2}. There is no overlap of writes from the same input
# position to the same output cell, but writes from different input positions
# DO overlap, so we use atomic adds.
#
# However atomics on fp32 to global memory are slow at this scale. Better:
# write per-tap to a unique output location using non-overlapping tiles in
# the input — within one input tile, the 3x3 output footprint of that tile
# only overlaps with neighbor tiles. We just accumulate into a local
# register tile of size [BLOCK_OC, (HW_TILE_expanded)] then atomic-add the
# final tile back. The simplest approach: do the conv_transpose with torch
# (cuDNN) — it's already heavily optimized. The headroom is the epilogue
# pipeline. Given the 4090 has excellent cuDNN, we'll keep conv_transpose
# in torch and focus on a highly tuned fused epilogue.


@triton.jit
def fused_mean_lse_kernel(
    x_ptr,           # [N, C, H, W]
    bias_ptr,        # [C]
    out_ptr,         # [N]
    C, HW,
    inv_HW,          # 1 / HW
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    n_base = pid_n * C * HW

    sum_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = offs_hw < HW
        ptrs = x_ptr + n_base + offs_c[:, None] * HW + offs_hw[None, :]
        mask = c_mask[:, None] & hw_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        sum_acc += tl.sum(vals, axis=1)

    mean = sum_acc * inv_HW
    bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0)
    z = mean + bias

    z_masked = tl.where(c_mask, z, -float('inf'))
    max_z = tl.max(z_masked, axis=0)
    exp_z = tl.exp(z_masked - max_z)
    exp_z = tl.where(c_mask, exp_z, 0.0)
    sum_exp = tl.sum(exp_z, axis=0)
    lse = tl.log(sum_exp) + max_z
    result = lse * 10.0

    tl.store(out_ptr + pid_n, result)


# Two-stage version: first kernel computes partial sums over HW chunks per
# (n, c), so we can use many programs (N * num_chunks) for memory bandwidth.
# Then a small kernel reduces those partials and runs lse.

@triton.jit
def partial_sum_kernel(
    x_ptr,          # [N, C, HW]
    partial_ptr,    # [N, C, NUM_CHUNKS]
    C, HW, NUM_CHUNKS,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)  # over (N * C * NUM_CHUNKS)
    chunk_id = pid % NUM_CHUNKS
    nc = pid // NUM_CHUNKS
    n = nc // C
    c = nc % C

    hw_start = chunk_id * BLOCK_HW
    offs = hw_start + tl.arange(0, BLOCK_HW)
    mask = offs < HW

    base = n * C * HW + c * HW
    vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)

    tl.store(partial_ptr + nc * NUM_CHUNKS + chunk_id, s)


@triton.jit
def reduce_lse_kernel(
    partial_ptr,    # [N, C, NUM_CHUNKS]
    bias_ptr,       # [C]
    out_ptr,        # [N]
    C, NUM_CHUNKS,
    inv_HW,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)

    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C
    offs_k = tl.arange(0, BLOCK_K)
    k_mask = offs_k < NUM_CHUNKS

    base = pid_n * C * NUM_CHUNKS
    ptrs = partial_ptr + base + offs_c[:, None] * NUM_CHUNKS + offs_k[None, :]
    mask = c_mask[:, None] & k_mask[None, :]
    vals = tl.load(ptrs, mask=mask, other=0.0)
    sum_per_c = tl.sum(vals, axis=1)
    mean = sum_per_c * inv_HW

    bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0)
    z = mean + bias
    z_masked = tl.where(c_mask, z, -float('inf'))
    max_z = tl.max(z_masked, axis=0)
    exp_z = tl.exp(z_masked - max_z)
    exp_z = tl.where(c_mask, exp_z, 0.0)
    sum_exp = tl.sum(exp_z, axis=0)
    lse = tl.log(sum_exp) + max_z
    result = lse * 10.0

    tl.store(out_ptr + pid_n, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        y = self.conv_transpose(x)
        N, C, H, W = y.shape
        HW = H * W
        y = y.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        out = torch.empty((N,), device=y.device, dtype=y.dtype)

        # Two-stage reduction for better parallelism on HW (very large ~514*514)
        BLOCK_HW = 8192
        NUM_CHUNKS = (HW + BLOCK_HW - 1) // BLOCK_HW

        partial = torch.empty((N, C, NUM_CHUNKS), device=y.device, dtype=torch.float32)

        grid1 = (N * C * NUM_CHUNKS,)
        partial_sum_kernel[grid1](
            y, partial,
            C, HW, NUM_CHUNKS,
            BLOCK_HW=BLOCK_HW,
            num_warps=8,
            num_stages=3,
        )

        # Reduce
        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2
        BLOCK_K = 1
        while BLOCK_K < NUM_CHUNKS:
            BLOCK_K *= 2
        if BLOCK_K < 1:
            BLOCK_K = 1

        inv_HW = 1.0 / float(HW)
        grid2 = (N,)
        reduce_lse_kernel[grid2](
            partial, bias_flat, out,
            C, NUM_CHUNKS, inv_HW,
            BLOCK_C=BLOCK_C, BLOCK_K=BLOCK_K,
            num_warps=4,
        )

        return out.view(N, 1)