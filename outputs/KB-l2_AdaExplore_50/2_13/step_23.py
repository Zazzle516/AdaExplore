import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_mean_bias_softmax_tanh_scale_kernel(
    x_ptr, bias_ptr, out_ptr,
    B, C, D, HW,
    inv_D,
    scaling_factor,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    b = tl.program_id(0)
    hw_block = tl.program_id(1)

    offs_hw = hw_block * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_hw = offs_hw < HW
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base_b = b * C * D * HW

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for d in range(0, D):
        ptrs = x_ptr + base_b + offs_c[:, None] * (D * HW) + d * HW + offs_hw[None, :]
        mask = mask_c[:, None] & mask_hw[None, :]
        vals = tl.load(ptrs, mask=mask, other=0.0)
        acc += vals

    mean = acc * inv_D

    bias = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    m = mean + bias[:, None]
    m = tl.where(mask_c[:, None], m, -float('inf'))

    mx = tl.max(m, axis=0)
    e = tl.exp(m - mx[None, :])
    e = tl.where(mask_c[:, None], e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s[None, :]

    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    out_base = b * C * HW
    out_ptrs = out_ptr + out_base + offs_c[:, None] * HW + offs_hw[None, :]
    store_mask = mask_c[:, None] & mask_hw[None, :]
    tl.store(out_ptrs, out, mask=store_mask)


def fused_mean_bias_softmax_tanh_scale(x, bias, scaling_factor):
    B, C, D, H, W = x.shape
    HW = H * W
    x = x.contiguous()
    bias_flat = bias.contiguous().view(C)
    out = torch.empty((B, C, 1, H, W), device=x.device, dtype=x.dtype)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_HW = 128
    grid = (B, (HW + BLOCK_HW - 1) // BLOCK_HW)
    fused_mean_bias_softmax_tanh_scale_kernel[grid](
        x, bias_flat, out,
        B, C, D, HW,
        1.0 / D,
        scaling_factor,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW,
        num_warps=8, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        out = fused_mean_bias_softmax_tanh_scale(x, self.bias, self.scaling_factor)
        return out