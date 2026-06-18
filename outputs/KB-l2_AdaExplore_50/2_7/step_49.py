import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr,
    n_elements, channel_stride, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c_idx = (offs // channel_stride) % n_channels
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

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

    tl.store(x_ptr + offs, out, mask=mask)


def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    channel_stride = D * H * W
    n_elements = x.numel()
    bias_flat = bias.contiguous().view(-1)

    BLOCK_SIZE = 8192
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_activation_bias_kernel[grid](
        x, bias_flat,
        n_elements, channel_stride, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
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