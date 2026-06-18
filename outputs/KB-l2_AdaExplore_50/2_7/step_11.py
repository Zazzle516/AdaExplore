import torch
import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def fused_activation_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, channel_stride, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    c_idx = (offsets // channel_stride) % n_channels
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # After ReLU: x >= 0, so LeakyReLU is a no-op.
    x = tl.maximum(x, 0.0)
    # GELU (tanh approx)
    k0 = 0.7978845608028654
    k1 = 0.044715
    inner = k0 * (x + k1 * x * x * x)
    e2 = tl.exp(2.0 * inner)
    tanh_val = (e2 - 1.0) / (e2 + 1.0)
    gelu = 0.5 * x * (1.0 + tanh_val)
    # Sigmoid
    sig = 1.0 / (1.0 + tl.exp(-gelu))
    out = sig + b

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_activation_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n_elements = x.numel()
    N, C, D, H, W = x.shape
    channel_stride = D * H * W
    bias_flat = bias.contiguous().view(-1)

    BLOCK_SIZE = 2048
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_activation_bias_kernel[grid](
        x, bias_flat, out,
        n_elements, channel_stride, C,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_activation_bias(x, self.bias)
        return x