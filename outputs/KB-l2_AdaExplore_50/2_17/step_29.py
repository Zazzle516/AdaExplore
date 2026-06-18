import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _instance_norm_div_kernel(
    x_ptr, out_ptr,
    HW,
    inv_HW,
    inv_divide,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW
    x_ptr = x_ptr + base
    out_ptr = out_ptr + base

    sum_x = 0.0
    sum_x2 = 0.0

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0)
        sum_x += tl.sum(v, axis=0)
        sum_x2 += tl.sum(v * v, axis=0)

    mean = sum_x * inv_HW
    mean_sq = sum_x2 * inv_HW
    var = mean_sq - mean * mean
    rstd = tl.rsqrt(var + eps)
    scale = rstd * inv_divide
    bias = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0)
        y = v * scale + bias
        tl.store(out_ptr + idx, y, mask=mask)


def instance_norm_div(x: torch.Tensor, divide_by: float, eps: float = 1e-5):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    grid = (N * C,)
    _instance_norm_div_kernel[grid](
        x, out,
        HW,
        1.0 / HW,
        1.0 / divide_by,
        eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = float(divide_by)

    def forward(self, x):
        x = self.conv(x)
        x = instance_norm_div(x, self.divide_by)
        return x