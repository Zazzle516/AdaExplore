import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _instance_norm_div_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, c)
    x_ptr = x_ptr + pid * HW
    out_ptr = out_ptr + pid * HW

    # First pass: compute sum and sum of squares
    sum_val = 0.0
    sum_sq = 0.0
    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.sum(v, axis=0)
        sum_sq += tl.sum(v * v, axis=0)

    mean = sum_val / HW
    var = sum_sq / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        y = v * scale + shift
        tl.store(out_ptr + idx, y, mask=mask)


def instance_norm_div(x, divide_by, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    x = x.contiguous()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = (N * C,)
    _instance_norm_div_kernel[grid](
        x, out, N, C, HW,
        1.0 / float(divide_by),
        float(eps),
        BLOCK=BLOCK,
        num_warps=4,
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