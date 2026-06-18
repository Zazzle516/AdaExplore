import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def partial_sum_kernel(
    x_ptr,
    partial_ptr,
    HW, NUM_CHUNKS,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk_id = pid % NUM_CHUNKS
    nc = pid // NUM_CHUNKS

    hw_start = chunk_id * BLOCK_HW
    offs = hw_start + tl.arange(0, BLOCK_HW)
    mask = offs < HW

    base = nc * HW
    vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    s = tl.sum(vals, axis=0)

    tl.store(partial_ptr + nc * NUM_CHUNKS + chunk_id, s)


@triton.jit
def reduce_lse_kernel(
    partial_ptr,
    bias_ptr,
    out_ptr,
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

        BLOCK_HW = 32768
        NUM_CHUNKS = (HW + BLOCK_HW - 1) // BLOCK_HW

        partial = torch.empty((N, C, NUM_CHUNKS), device=y.device, dtype=torch.float32)

        grid1 = (N * C * NUM_CHUNKS,)
        partial_sum_kernel[grid1](
            y, partial,
            HW, NUM_CHUNKS,
            BLOCK_HW=BLOCK_HW,
            num_warps=8,
            num_stages=4,
        )

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