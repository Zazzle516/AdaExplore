import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish_tanh_kernel_cl(
    x_ptr, bias_ptr, out_ptr, n_elements, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channels_last: channel is the last (contiguous) dim
    ch = offsets % C
    b = tl.load(bias_ptr + ch, mask=mask, other=0.0)
    x = x + b
    e = tl.exp(x)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + e))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    mish = x * tanh_sp
    out = 2.0 * tl.sigmoid(2.0 * mish) - 1.0
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def _mish_tanh_kernel(
    x_ptr, bias_ptr, out_ptr, n_elements, C, spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    ch = (offsets // spatial) % C
    b = tl.load(bias_ptr + ch, mask=mask, other=0.0)
    x = x + b
    e = tl.exp(x)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + e))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    mish = x * tanh_sp
    out = 2.0 * tl.sigmoid(2.0 * mish) - 1.0
    tl.store(out_ptr + offsets, out, mask=mask)


def mish_tanh_bias_cl(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    C = x.shape[1]
    BLOCK_SIZE = 8192
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _mish_tanh_kernel_cl[grid](x, bias, out, n, C, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        # Convert weight to channels_last_3d for faster cuDNN algo selection
        self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
        self._stride = stride
        self._padding = padding

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        y = F.conv3d(x, self.conv.weight, bias=None, stride=self._stride, padding=self._padding)
        # y is in channels_last_3d. Need contiguous access pattern matching layout.
        # In channels_last_3d, memory order is N, D, H, W, C; channel is innermost.
        # We allocate output with same layout and index channel as offset % C.
        return mish_tanh_bias_cl(y, self.conv.bias)