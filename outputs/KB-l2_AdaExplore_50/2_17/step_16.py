import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def instance_norm_div_kernel_single(
    x_ptr, out_ptr,
    HW,
    inv_hw,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW
    idx = tl.arange(0, BLOCK)
    mask = idx < HW
    x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
    xf = x.to(tl.float32)
    xz = tl.where(mask, xf, 0.0)
    sum_x = tl.sum(xz, axis=0)
    sum_x2 = tl.sum(xz * xz, axis=0)
    mean = sum_x * inv_hw
    var = sum_x2 * inv_hw - mean * mean
    rstd = tl.rsqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale
    y = xf * scale + shift
    tl.store(out_ptr + base + idx, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK': 32768}, num_warps=16, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def instance_norm_div_kernel(
    x_ptr, out_ptr,
    HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * HW

    sum_x = 0.0
    sum_x2 = 0.0
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_x += tl.sum(x, axis=0)
        sum_x2 += tl.sum(x * x, axis=0)

    inv_hw = 1.0 / HW
    mean = sum_x * inv_hw
    var = sum_x2 * inv_hw - mean * mean
    rstd = tl.rsqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * scale + shift
        tl.store(out_ptr + base + idx, y, mask=mask)


def instance_norm_div(x: torch.Tensor, divide_by: float, eps: float = 1e-5):
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    # Pick single-pass kernel when HW fits in a single block
    if HW <= 16384:
        BLOCK = 16384
        instance_norm_div_kernel_single[grid](
            x, out,
            HW,
            1.0 / HW,
            1.0 / divide_by,
            eps,
            BLOCK=BLOCK,
            num_warps=8,
            num_stages=2,
        )
    elif HW <= 32768:
        BLOCK = 32768
        instance_norm_div_kernel_single[grid](
            x, out,
            HW,
            1.0 / HW,
            1.0 / divide_by,
            eps,
            BLOCK=BLOCK,
            num_warps=16,
            num_stages=2,
        )
    else:
        instance_norm_div_kernel[grid](
            x, out,
            HW,
            1.0 / divide_by,
            eps,
        )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.instance_norm = nn.InstanceNorm2d(out_channels)
        self.divide_by = float(divide_by)

    def forward(self, x):
        x = self.conv(x)
        x = instance_norm_div(x, self.divide_by, eps=1e-5)
        return x