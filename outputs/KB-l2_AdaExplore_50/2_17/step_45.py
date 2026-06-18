import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=32, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit
def _instance_norm_div_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    inv_divide,
    inv_HW,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, c)
    x_ptr = x_ptr + pid * HW
    out_ptr = out_ptr + pid * HW

    idx = tl.arange(0, BLOCK)
    mask = idx < HW
    v = tl.load(x_ptr + idx, mask=mask, other=0.0)

    sum_x = tl.sum(v, axis=0)
    sum_x2 = tl.sum(v * v, axis=0)

    mean = sum_x * inv_HW
    mean_sq = sum_x2 * inv_HW
    var = mean_sq - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_divide
    bias = -mean * scale

    y = v * scale + bias
    tl.store(out_ptr + idx, y, mask=mask)


def instance_norm_div(x: torch.Tensor, divide_by: float, eps: float = 1e-5):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    BLOCK = triton.next_power_of_2(HW)

    grid = (N * C,)
    _instance_norm_div_kernel[grid](
        x, x,
        N, C, HW,
        1.0 / divide_by,
        1.0 / HW,
        eps,
        BLOCK=BLOCK,
    )
    return x


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = float(divide_by)

    def forward(self, x):
        x = self.conv(x)
        x = instance_norm_div(x, self.divide_by)
        return x