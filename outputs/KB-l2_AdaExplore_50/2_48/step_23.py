import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    S, total,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # channel index from linear offset: offs // S % C
    c = (offs // S) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    y = x * s
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)
    z = t * b
    out = 1.0 / (1.0 + tl.exp(-z))
    tl.store(out_ptr + offs, out, mask=mask)


def fused_epilogue(x, scale, bias):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    total = N * C * S
    BLOCK = 4096
    grid = ((total + BLOCK - 1) // BLOCK,)
    fused_epilogue_kernel[grid](
        x_c, scale.contiguous().view(-1), bias.contiguous().view(-1), out,
        S, total,
        C=C,
        BLOCK=BLOCK,
        num_warps=8,
        num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(ModelNew, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        return fused_epilogue(x, self.scaling_factor, self.bias)