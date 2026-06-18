import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# We need to perform the conv_transpose to get the full output, then
# global avg pool, etc. The safety contract says we must materialize
# the full output of the heavy op. So we do the conv_transpose with
# torch (or a custom kernel), but we can fuse the mean + bias + logsumexp + sum + mul.

# Strategy: do conv_transpose with PyTorch (cuDNN), then a fused Triton kernel
# that computes mean over H,W per (n, c), adds bias, logsumexp over channels,
# multiplies by 10. Sum over (2,3) is trivial since after mean+keepdim it's 1x1.

@triton.jit
def fused_post_kernel(
    x_ptr,           # [N, C, H, W] conv_transpose output
    bias_ptr,        # [C]
    out_ptr,         # [N, 1]
    N, C, H, W,
    HW,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per batch
    if pid >= N:
        return

    # Step 1: compute mean over H,W for each channel -> means[C]
    # Step 2: add bias -> z[C] = mean[c] + bias[c]
    # Step 3: logsumexp over c -> scalar
    # Step 4: multiply by 10

    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < C

    # Compute mean for each channel using a loop over HW chunks
    # We'll compute one channel at a time? No - vectorize over channels.
    # For each chunk of HW, accumulate sum across all channels.
    # x layout: [N, C, H, W], stride for n = C*H*W, for c = H*W

    sum_acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    n_offset = pid * C * HW

    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = offs_hw < HW
        # ptrs: [BLOCK_C, BLOCK_HW]
        ptrs = x_ptr + n_offset + offs_c[:, None] * HW + offs_hw[None, :]
        mask = c_mask[:, None] & hw_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        sum_acc += tl.sum(vals, axis=1)

    mean = sum_acc / HW

    bias = tl.load(bias_ptr + offs_c, mask=c_mask, other=0.0)
    z = mean + bias

    # logsumexp over c
    z_masked = tl.where(c_mask, z, -float('inf'))
    max_z = tl.max(z_masked, axis=0)
    exp_z = tl.exp(z_masked - max_z)
    exp_z = tl.where(c_mask, exp_z, 0.0)
    sum_exp = tl.sum(exp_z, axis=0)
    lse = tl.log(sum_exp) + max_z

    result = lse * 10.0
    tl.store(out_ptr + pid, result)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.out_channels = out_channels

    def forward(self, x):
        x = self.conv_transpose(x)
        N, C, H, W = x.shape
        HW = H * W

        x = x.contiguous()
        bias_flat = self.bias.view(-1).contiguous()

        out = torch.empty((N, 1), device=x.device, dtype=x.dtype)

        # Choose BLOCK_C as next power of 2 >= C
        BLOCK_C = 1
        while BLOCK_C < C:
            BLOCK_C *= 2

        BLOCK_HW = 16384

        grid = (N,)
        fused_post_kernel[grid](
            x, bias_flat, out,
            N, C, H, W, HW,
            BLOCK_HW=BLOCK_HW,
            BLOCK_C=BLOCK_C,
            num_warps=16,
            num_stages=2,
        )

        return out