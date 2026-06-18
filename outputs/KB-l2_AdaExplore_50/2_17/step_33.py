import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK': 4096}, num_warps=16, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _instance_norm_div_kernel(
    x_ptr, out_ptr, bias_ptr,
    N, C, HW,
    inv_HW,
    inv_divide,
    eps,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (n, c)
    c = pid % C
    x_ptr = x_ptr + pid * HW
    out_ptr = out_ptr + pid * HW
    b = tl.load(bias_ptr + c).to(tl.float32)

    sum_x = tl.zeros([BLOCK], dtype=tl.float32)
    sum_x2 = tl.zeros([BLOCK], dtype=tl.float32)

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32) + b
        sum_x += tl.where(mask, v, 0.0)
        sum_x2 += tl.where(mask, v * v, 0.0)

    mean = tl.sum(sum_x, axis=0) * inv_HW
    mean_sq = tl.sum(sum_x2, axis=0) * inv_HW
    var = mean_sq - mean * mean
    rstd = 1.0 / tl.sqrt(var + eps)
    scale = rstd * inv_divide
    bias_term = -mean * scale

    for off in range(0, HW, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        mask = idx < HW
        v = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32) + b
        y = v * scale + bias_term
        tl.store(out_ptr + idx, y, mask=mask)


def instance_norm_div(x: torch.Tensor, bias: torch.Tensor, divide_by: float, eps: float = 1e-5):
    assert x.is_cuda and x.dtype == torch.float32
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    grid = (N * C,)
    _instance_norm_div_kernel[grid](
        x, out, bias,
        N, C, HW,
        1.0 / HW,
        1.0 / divide_by,
        eps,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, divide_by):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Keep conv bias separately and disable conv's bias add (fused into norm kernel)
        self.bias = nn.Parameter(conv.bias.detach().clone())
        conv.bias = None
        self.conv = conv
        self.divide_by = float(divide_by)

    def forward(self, x):
        x = self.conv(x)
        x = instance_norm_div(x, self.bias, self.divide_by)
        return x