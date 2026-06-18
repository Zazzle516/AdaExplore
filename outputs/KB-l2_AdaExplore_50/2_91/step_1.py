import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = n * C * HW + hw
    x_ptrs = x_ptr + base + offs_c * HW

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))
    x_f = x.to(tl.float32)

    max_val = tl.max(x_f, axis=0)
    e = tl.exp(x_f - max_val)
    e = tl.where(mask_c, e, 0.0)
    sum_e = tl.sum(e, axis=0)
    sm = e / sum_e

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = (sm + b) * scaling_factor
    out = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c * HW
    tl.store(out_ptrs, out.to(x.dtype), mask=mask_c)


def fused_softmax_bias_scale_sigmoid(x, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * HW,)

    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias, out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        bias_flat = self.bias.view(-1).contiguous()
        x = x.contiguous()
        return fused_softmax_bias_scale_sigmoid(x, bias_flat, self.scaling_factor)