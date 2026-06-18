import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel_nhwc(
    x_ptr, bias_ptr, out_ptr,
    C,
    SCALING: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    base = pid * C
    x_ptrs = x_ptr + base + offs_c

    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))
    x_f = x.to(tl.float32)

    max_val = tl.max(x_f, axis=0)
    e = tl.exp(x_f - max_val)
    e = tl.where(mask_c, e, 0.0)
    sum_e = tl.sum(e, axis=0)
    sm = e / sum_e

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = (sm + b) * SCALING
    out = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c
    tl.store(out_ptrs, out.to(x.dtype), mask=mask_c)


def fused_softmax_bias_scale_sigmoid(x, bias, scaling_factor):
    # x is in channels_last layout: shape (N,C,H,W), stride (C*H*W,1,C*W,C)
    N, C, H, W = x.shape
    out = torch.empty_like(x)

    BLOCK_C = triton.next_power_of_2(C)
    grid = (N * H * W,)

    fused_softmax_bias_scale_sigmoid_kernel_nhwc[grid](
        x, bias, out,
        C,
        SCALING=float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=2,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous(memory_format=torch.channels_last)
        bias_flat = self.bias.view(-1).contiguous()
        return fused_softmax_bias_scale_sigmoid(x, bias_flat, self.scaling_factor)