import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
    ],
    key=['S', 'C'],
)
@triton.jit
def fused_kernel(
    x_ptr, bias_ptr, out_ptr,
    C, S,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)

    c = pid_nc % C
    b = tl.load(bias_ptr + c)

    base = pid_nc * S
    offs = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < S

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    y = (2.0 * x + b) * x + x
    tl.store(out_ptr + base + offs, y, mask=mask)


def fused_op(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    N, C, D, H, W = x.shape
    S = D * H * W
    out = torch.empty_like(x)
    bias_flat = bias.contiguous().view(-1)
    grid = lambda meta: (N * C, (S + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])
    fused_kernel[grid](
        x, bias_flat, out,
        C, S,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_op(x, self.bias)