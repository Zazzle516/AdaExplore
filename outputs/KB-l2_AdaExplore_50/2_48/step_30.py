import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    S,
    C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    c = pid_nc % C

    offs_s = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs_s < S

    base = pid_nc * S
    x = tl.load(x_ptr + base + offs_s, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c)
    b = tl.load(bias_ptr + c)

    y = x * s
    t = 2.0 * tl.sigmoid(2.0 * y) - 1.0
    z = t * b
    out = tl.sigmoid(z)
    tl.store(out_ptr + base + offs_s, out, mask=mask)


def fused_epilogue(x, scale, bias):
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    BLOCK = 4096
    grid = (N * C, (S + BLOCK - 1) // BLOCK)
    fused_epilogue_kernel[grid](
        x, scale.contiguous().view(-1), bias.contiguous().view(-1), out,
        S,
        C=C,
        BLOCK=BLOCK,
        num_warps=4,
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
        if not x.is_contiguous():
            x = x.contiguous()
        return fused_epilogue(x, self.scaling_factor, self.bias)