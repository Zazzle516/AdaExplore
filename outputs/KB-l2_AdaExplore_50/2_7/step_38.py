import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr,
    channel_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_s = tl.program_id(2)

    base = (pid_n * tl.num_programs(1) + pid_c) * channel_stride
    offs = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < channel_stride
    ptrs = x_ptr + base + offs

    x = tl.load(ptrs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + pid_c)

    # ReLU (LeakyReLU is no-op after ReLU since x>=0)
    x = tl.maximum(x, 0.0)
    # GELU (tanh approx)
    k0 = 0.7978845608028654
    k1 = 0.044715
    inner = k0 * (x + k1 * x * x * x)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_val)
    sig = tl.sigmoid(gelu)
    out = sig + b

    tl.store(ptrs, out, mask=mask)


def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    channel_stride = D * H * W
    bias_flat = bias.contiguous().view(-1)

    BLOCK_SIZE = 4096
    grid = (N, C, (channel_stride + BLOCK_SIZE - 1) // BLOCK_SIZE)
    fused_activation_bias_kernel[grid](
        x, bias_flat,
        channel_stride,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_bias(x, self.bias)
        return x