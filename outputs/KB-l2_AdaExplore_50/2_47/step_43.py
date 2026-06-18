import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _mish_tanh_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # softplus, reusing exp(x)
    e = tl.exp(x)
    sp = tl.where(x > 20.0, x, tl.log(1.0 + e))
    # tanh(sp) = 2*sigmoid(2*sp) - 1
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    mish = x * tanh_sp
    # tanh(mish) = 2*sigmoid(2*mish) - 1
    out = 2.0 * tl.sigmoid(2.0 * mish) - 1.0
    tl.store(out_ptr + offsets, out, mask=mask)


def mish_tanh(x: torch.Tensor) -> torch.Tensor:
    if not x.is_contiguous(memory_format=torch.channels_last_3d) and not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 8192
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _mish_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv = self.conv.to(memory_format=torch.channels_last_3d)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.conv(x)
        x = mish_tanh(x)
        return x