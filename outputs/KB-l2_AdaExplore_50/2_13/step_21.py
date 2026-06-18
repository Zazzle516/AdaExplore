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
    pid = tl.program_id(0)
    b = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = tl.arange(0, BLOCK_D)

    base = b * C * D * HW + hw
    ptrs = x_ptr + base + offs_c[:, None] * D * HW + offs_d[None, :] * HW
    vals = tl.load(ptrs)
    sum_d = tl.sum(vals, axis=1)
    mean_d = sum_d / D

    bias = tl.load(bias_ptr + offs_c)
    m = mean_d + bias

    mx = tl.max(m, axis=0)
    e = tl.exp(m - mx)
    s = tl.sum(e, axis=0)
    sm = e / s

    e2 = tl.exp(2.0 * sm)
    t = (e2 - 1.0) / (e2 + 1.0)
    out = t * scaling_factor

    out_ptrs = out_ptr + b * C * HW + offs_c * HW + hw
    tl.store(out_ptrs, out)


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
        num_warps=2, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last_3d)
        self.bias = nn.Parameter(torch.randn(1, out_channels, 1, 1, 1))
        self.scaling_factor = scaling_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self._bias_flat = None

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv_transpose(x)  # (B, C, D, H, W)
        x = x.contiguous()
        if self._bias_flat is None or self._bias_flat.device != self.bias.device:
            self._bias_flat = self.bias.view(self.out_channels)
        out = fused_mean_bias_softmax_tanh_scale(x, self._bias_flat, self.scaling_factor)
        return out