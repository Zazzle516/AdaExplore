import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


# The key insight: after conv_transpose, we take global average pooling over H,W.
# Global mean of conv_transpose output for each (n, oc) channel.
# We MUST compute the full conv_transpose output per the safety contract.
# 
# However we can still use a custom kernel for the conv_transpose + mean fusion,
# computing the full conv but accumulating directly into a (N, OC) mean tensor.
# This still does the full O(N*OC*IC*H*W*KH*KW) work.
#
# Actually the simpler approach: just use torch's conv_transpose2d (cuDNN is fast),
# then fuse mean + bias + logsumexp + sum + mul into a single kernel.

@triton.jit
def fused_post_kernel(
    x_ptr,          # conv output: (N, OC, H, W)
    bias_ptr,       # (OC,)
    out_ptr,        # (N,)
    N, OC, H, W,
    HW,
    inv_hw,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    n = tl.program_id(0)
    # Compute mean per channel for this n
    # offsets in OC
    oc_offs = tl.arange(0, BLOCK_OC)
    oc_mask = oc_offs < OC
    
    # We'll compute per-channel mean by iterating
    # acc shape: (BLOCK_OC,)
    acc = tl.zeros((BLOCK_OC,), dtype=tl.float32)
    
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offs = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offs < HW
        # ptr: x[n, oc, hw] = x_ptr + n*OC*HW + oc*HW + hw
        ptrs = x_ptr + n * OC * HW + oc_offs[:, None] * HW + hw_offs[None, :]
        mask = oc_mask[:, None] & hw_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=1)
    
    mean_vals = acc * inv_hw  # (BLOCK_OC,)
    
    # Add bias
    bias_vals = tl.load(bias_ptr + oc_offs, mask=oc_mask, other=0.0)
    z = mean_vals + bias_vals  # (BLOCK_OC,)
    
    # logsumexp over oc dimension (only valid entries)
    # mask invalid to -inf
    neg_inf_val = float('-inf')
    z_masked = tl.where(oc_mask, z, neg_inf_val)
    max_z = tl.max(z_masked, axis=0)
    exp_z = tl.exp(z_masked - max_z)
    exp_z = tl.where(oc_mask, exp_z, 0.0)
    sum_exp = tl.sum(exp_z, axis=0)
    lse = max_z + tl.log(sum_exp)
    
    # sum over (2,3) - but after mean keepdim, H=W=1, so sum is identity
    # multiply by 10
    result = lse * 10.0
    
    tl.store(out_ptr + n, result)


def fused_post(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    x: (N, OC, H, W)
    bias: (OC, 1, 1) or (OC,)
    returns: (N, 1)
    """
    N, OC, H, W = x.shape
    HW = H * W
    bias_flat = bias.reshape(-1).contiguous()
    out = torch.empty(N, device=x.device, dtype=torch.float32)
    
    # Pick BLOCK_OC as next power of 2 >= OC
    BLOCK_OC = 1
    while BLOCK_OC < OC:
        BLOCK_OC *= 2
    
    grid = (N,)
    fused_post_kernel[grid](
        x, bias_flat, out,
        N, OC, H, W, HW,
        1.0 / HW,
        BLOCK_OC=BLOCK_OC,
        BLOCK_HW=1024,
        num_warps=8,
    )
    
    return out.view(N, 1)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        x = x.contiguous()
        return fused_post(x, self.bias)