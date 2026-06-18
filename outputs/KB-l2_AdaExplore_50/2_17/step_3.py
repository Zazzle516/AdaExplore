import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def instance_norm_div_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    inv_div,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    base = n * C * HW + c * HW

    # First pass: compute mean and var via sum/sum_sq
    sum_val = tl.zeros([BLOCK], dtype=tl.float32)
    sum_sq = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        sum_val += tl.where(mask, x, 0.0)
        sum_sq += tl.where(mask, x * x, 0.0)

    s = tl.sum(sum_val, axis=0)
    sq = tl.sum(sum_sq, axis=0)
    mean = s / HW
    var = sq / HW - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_div
    shift = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0).to(tl.float32)
        y = x * scale + shift
        tl.store(out_ptr + base + idx, y, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divide_by = float(divide_by)
        self.eps = 1e-5

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W = x.shape
        HW = H * W
        x_contig = x.contiguous()
        out = torch.empty_like(x_contig)

        BLOCK = 1024
        grid = (N * C,)
        instance_norm_div_kernel[grid](
            x_contig, out,
            N, C, HW,
            1.0 / self.divide_by,
            self.eps,
            BLOCK=BLOCK,
            num_warps=4,
        )
        return out