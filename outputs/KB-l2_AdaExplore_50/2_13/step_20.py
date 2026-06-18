import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_mean_bias_softmax_tanh_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # grid: (B * HW,)
    # x: (B, C, D, HW) — we treat H*W flattened
    # For each (b, hw), compute over all channels:
    #   m[c] = mean_d x[b,c,d,hw] + bias[c]
    # Then softmax over c, tanh, scale.
    pid = tl.program_id(0)
    b = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Compute per-channel mean over depth
    # x layout: (B, C, D, HW); offset = b*C*D*HW + c*D*HW + d*HW + hw
    # For vector over c: base + c*D*HW + d*HW (2D over c, d)
    base = b * C * D * HW + hw
    # 2D pointers: [BLOCK_C, BLOCK_D]
    ptrs = x_ptr + base + offs_c[:, None] * D * HW + offs_d[None, :] * HW
    mask = mask_c[:, None] & mask_d[None, :]
    vals = tl.load(ptrs, mask=mask, other=0.0)
    # Sum over depth
    sum_d = tl.sum(vals, axis=1)  # [BLOCK_C]
    mean_d = sum_d / D

    # Add bias
    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    m = mean_d + bias
    m = tl.where(mask_c, m, -float('inf'))

    # Softmax
    mx = tl.max(m, axis=0)
    e = tl.exp(m - mx)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # tanh + scale
    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    # Output shape: (B, C, 1, HW), offset = b*C*HW + c*HW + hw
    out_ptrs = out_ptr + b * C * HW + offs_c * HW + hw
    tl.store(out_ptrs, out, mask=mask_c)


def fused_mean_bias_softmax_tanh_scale(x, bias, scaling_factor):
    # x: (B, C, D, H, W); bias: (1, C, 1, 1, 1) -> we'll pass as (C,)
    B, C, D, H, W = x.shape
    HW = H * W
    x = x.contiguous()
    bias_flat = bias.contiguous().view(C)
    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)
    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_D = triton.next_power_of_2(D)
    grid = (B * HW,)
    fused_mean_bias_softmax_tanh_scale_kernel[grid](
        x, bias_flat, out,
        B, C, D, HW,
        scaling_factor,
        BLOCK_C=BLOCK_C, BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        # Use cudnn conv_transpose3d (much faster than a custom Triton impl)
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        # Fused mean(dim=2) + bias add + softmax(dim=1) + tanh + scale
        out = fused_mean_bias_softmax_tanh_scale(x, self.bias, self.scaling_factor)
        return out