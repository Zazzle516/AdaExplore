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
    # mish: x * tanh(softplus(x))
    sp = tl.log(1.0 + tl.exp(x))
    # tanh via sigmoid: tanh(z) = 2*sigmoid(2z) - 1
    t1 = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    m = x * t1
    t2 = 2.0 * tl.sigmoid(2.0 * m) - 1.0
    tl.store(out_ptr + offsets, t2, mask=mask)


def fused_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK = 16384
    grid = ((n + BLOCK - 1) // BLOCK,)
    _mish_tanh_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK, num_warps=4, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_mish_tanh(x)
        return x