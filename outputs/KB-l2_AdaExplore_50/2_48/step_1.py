import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    N, C, D, H, W,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    total = N * C * D * H * W
    mask = offs < total

    # compute channel index
    dhw = D * H * W
    c_idx = (offs // dhw) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    s = tl.load(scale_ptr + c_idx, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # tanh via exp
    v = x * s
    # tanh(v) = (exp(2v) - 1) / (exp(2v) + 1)
    e = tl.exp(2.0 * v)
    t = (e - 1.0) / (e + 1.0)
    y = t * b
    # sigmoid
    out = 1.0 / (1.0 + tl.exp(-y))

    tl.store(out_ptr + offs, out, mask=mask)


def fused_epilogue(x, scale, bias):
    x = x.contiguous()
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    total = x.numel()
    BLOCK = 1024
    grid = ((total + BLOCK - 1) // BLOCK,)
    fused_epilogue_kernel[grid](
        x, scale.contiguous().view(-1), bias.contiguous().view(-1), out,
        N, C, D, H, W, BLOCK=BLOCK,
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
        return fused_epilogue(x, self.scaling_factor, self.bias)