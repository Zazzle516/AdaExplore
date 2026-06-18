import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr,         # input: (N, C, H, W) after conv_transpose
    bias_ptr,      # bias: (C,)
    out_ptr,       # output: (N, C, H, W)
    N, C, H, W,
    scaling_factor,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, h, w) - reduce over C
    pid = tl.program_id(0)
    HW = H * W
    nhw = N * HW
    if pid >= nhw:
        return

    n = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W

    # base offset for this (n, h, w) across channels
    # offset(n, c, h, w) = n*C*H*W + c*H*W + h*W + w
    base = n * C * H * W + h * W + w
    stride_c = H * W

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    x_ptrs = x_ptr + base + offs_c * stride_c
    x = tl.load(x_ptrs, mask=mask_c, other=-float('inf'))

    # softmax along C
    m = tl.max(x, axis=0)
    x_shift = x - m
    e = tl.exp(x_shift)
    e = tl.where(mask_c, e, 0.0)
    s = tl.sum(e, axis=0)
    sm = e / s

    # add bias (per channel), scale, sigmoid
    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0)
    y = (sm + b) * scaling_factor
    y = 1.0 / (1.0 + tl.exp(-y))

    out_ptrs = out_ptr + base + offs_c * stride_c
    tl.store(out_ptrs, y, mask=mask_c)


def fused_softmax_bias_scale_sigmoid(x: torch.Tensor, bias: torch.Tensor, scaling_factor: float):
    assert x.is_cuda and bias.is_cuda
    x = x.contiguous()
    N, C, H, W = x.shape
    out = torch.empty_like(x)

    # bias is (C, 1, 1) -> flatten to (C,)
    bias_flat = bias.contiguous().view(-1)

    # next power of 2 for BLOCK_C
    BLOCK_C = 1
    while BLOCK_C < C:
        BLOCK_C *= 2

    grid = (N * H * W,)
    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias_flat, out,
        N, C, H, W,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        num_warps=4,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_softmax_bias_scale_sigmoid(x, self.bias, self.scaling_factor)
        return x