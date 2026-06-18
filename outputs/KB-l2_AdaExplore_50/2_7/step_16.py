import torch
import torch.nn as nn
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
    ],
    key=['channel_stride'],
)
@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    channel_stride, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)  # over N*C
    pid_s = tl.program_id(1)   # over spatial tiles within one channel

    c_idx = pid_nc % n_channels
    b = tl.load(bias_ptr + c_idx)

    base = pid_nc * channel_stride
    offs = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < channel_stride
    ptrs = x_ptr + base + offs
    x = tl.load(ptrs, mask=mask, other=0.0)

    # ReLU
    x = tl.maximum(x, 0.0)
    # LeakyReLU
    x = tl.where(x >= 0.0, x, x * 0.01)
    # GELU (tanh approximation)
    k0 = 0.7978845608028654
    k1 = 0.044715
    inner = k0 * (x + k1 * x * x * x)
    e_pos = tl.exp(inner)
    e_neg = tl.exp(-inner)
    tanh_val = (e_pos - e_neg) / (e_pos + e_neg)
    gelu = 0.5 * x * (1.0 + tanh_val)
    # Sigmoid
    sig = 1.0 / (1.0 + tl.exp(-gelu))
    out = sig + b

    tl.store(out_ptr + base + offs, out, mask=mask)


def fused_activation_bias(x: torch.Tensor, bias_flat: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    N, C, D, H, W = x.shape
    channel_stride = D * H * W

    grid = lambda META: (N * C, (channel_stride + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE'])
    fused_activation_bias_kernel[grid](
        x, bias_flat, out,
        channel_stride, C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        bias_flat = self.bias.view(-1)
        x = fused_activation_bias(x, bias_flat)
        return x