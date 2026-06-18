import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    N, C, S,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_nc = tl.program_id(1)
    n = pid_nc // C
    c = pid_nc % C

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < S

    base = (n * C + c) * S
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c)
    b = tl.load(bias_ptr + c)

    y = x * s
    # tanh via exp
    e2 = tl.exp(2.0 * y)
    t = (e2 - 1.0) / (e2 + 1.0)
    z = t * b
    out = 1.0 / (1.0 + tl.exp(-z))

    tl.store(out_ptr + base + offs, out, mask=mask)


def fused_epilogue(x, scale, bias):
    N, C, D, H, W = x.shape
    S = D * H * W
    x_c = x.contiguous()
    out = torch.empty_like(x_c)
    scale_flat = scale.contiguous().view(-1)
    bias_flat = bias.contiguous().view(-1)
    BLOCK = 1024
    grid = ((S + BLOCK - 1) // BLOCK, N * C)
    fused_epilogue_kernel[grid](
        x_c, scale_flat, bias_flat, out,
        N, C, S,
        BLOCK=BLOCK,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.scaling_factor, self.bias)
        return x