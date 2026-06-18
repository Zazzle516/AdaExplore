import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=3),
    ],
    key=['HW'],
)
@triton.jit
def instance_norm_div_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, c)
    base = pid * HW

    # Pass 1: compute mean and variance using online reduction
    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    mean = sum_x / HW
    var = sum_x2 / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    # Pass 2: normalize and divide
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * scale + shift
        tl.store(out_ptr + base + idx, y, mask=mask)


def instance_norm_div(x: torch.Tensor, divide_by: float, eps: float = 1e-5):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    grid = (N * C,)
    instance_norm_div_kernel[grid](
        x, out,
        N, C, HW,
        1.0 / divide_by,
        eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = divide_by

    def forward(self, x):
        x = self.conv(x)
        x = instance_norm_div(x, self.divide_by, eps=1e-5)
        return x