import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _instance_norm_div_kernel(
    x_ptr, out_ptr,
    HW,
    inv_div,
    inv_hw,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    x_ptr = x_ptr + pid * HW
    out_ptr = out_ptr + pid * HW

    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        idx = tl.max_contiguous(tl.multiple_of(idx, BLOCK), BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val * inv_hw
    var = sum_sq * inv_hw - mean * mean
    rstd = tl.rsqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        idx = tl.max_contiguous(tl.multiple_of(idx, BLOCK), BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = v * scale + shift
        tl.store(out_ptr + idx, y, mask=mask)


def instance_norm_div(x, divide_by, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    grid = (N * C,)
    _instance_norm_div_kernel[grid](
        x, out, HW,
        1.0 / float(divide_by),
        1.0 / float(HW),
        float(eps),
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
        x = instance_norm_div(x, self.divide_by)
        return x