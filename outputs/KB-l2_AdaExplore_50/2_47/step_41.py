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
    # mish: x * tanh(softplus(x)) = x * tanh(log(1+exp(x)))
    # use stable softplus
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    # tanh via formula
    e1 = tl.exp(sp)
    e2 = tl.exp(-sp)
    tanh_sp = (e1 - e2) / (e1 + e2)
    mish = x * tanh_sp
    # final tanh
    e3 = tl.exp(mish)
    e4 = tl.exp(-mish)
    out = (e3 - e4) / (e3 + e4)
    tl.store(out_ptr + offsets, out, mask=mask)


def mish_tanh(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _mish_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = mish_tanh(x)
        return x