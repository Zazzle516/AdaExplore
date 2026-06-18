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
    # numerically stable softplus: max(x,0) + log(1+exp(-|x|))
    ax = tl.abs(x)
    zero = ax - ax
    pos = tl.where(x > 0, x, zero)
    sp = pos + tl.log(1.0 + tl.exp(-ax))
    # tanh(sp) via 2*sigmoid(2*sp) - 1
    t1 = 2.0 / (1.0 + tl.exp(-2.0 * sp)) - 1.0
    m = x * t1
    # tanh(m) via 2*sigmoid(2*m) - 1
    t2 = 2.0 / (1.0 + tl.exp(-2.0 * m)) - 1.0
    tl.store(out_ptr + offsets, t2, mask=mask)


def mish_tanh(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    _mish_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = mish_tanh(x)
        return x