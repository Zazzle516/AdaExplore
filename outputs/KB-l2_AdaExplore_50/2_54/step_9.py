import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
    ],
    key=['HW', 'C'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, mult_ptr, out_ptr,
    n_elements, HW, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c = (offs // HW) % C
    m = tl.load(mult_ptr + c, mask=mask, other=0.0)

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    y = x * m
    y = tl.where(y >= 0, y, y * 0.01)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * y * (1.0 + tl.erf(y * inv_sqrt2))

    tl.store(out_ptr + offs, gelu, mask=mask)


def fused_epilogue(x, multiplier):
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_epilogue_kernel[grid](
        x, multiplier.contiguous().view(-1), out,
        n_elements, HW, C,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.multiplier)
        return x