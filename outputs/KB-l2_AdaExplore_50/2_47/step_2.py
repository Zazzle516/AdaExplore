import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_act_kernel(
    x_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # softplus(x) = log(1+exp(x)); tanh(sp) = 2*sigmoid(2*sp) - 1
    sp = tl.log(1.0 + tl.exp(x))
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    mish_val = x * th
    out = 2.0 * tl.sigmoid(2.0 * mish_val) - 1.0
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_mish_tanh(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    BLOCK_SIZE = 4096
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_act_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        x = fused_mish_tanh(x)
        return x