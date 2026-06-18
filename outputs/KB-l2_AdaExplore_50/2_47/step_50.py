import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
    ],
    key=['nc', 'spatial'],
)
@triton.jit
def _mish_tanh_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    spatial, num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    # 2D grid: pid_nc indexes (n*C), pid_s indexes spatial block
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    c = pid_nc % num_channels
    b = tl.load(bias_ptr + c)
    base = pid_nc * spatial
    offs = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < spatial
    addr = base + offs
    x = tl.load(x_ptr + addr, mask=mask, other=0.0)
    x = x + b
    # mish: x * tanh(softplus(x))
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    mish = x * tanh_sp
    out = 2.0 * tl.sigmoid(2.0 * mish) - 1.0
    tl.store(out_ptr + addr, out, mask=mask)


def mish_tanh_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    N = x.shape[0]
    C = x.shape[1]
    spatial = x.shape[2] * x.shape[3] * x.shape[4]
    nc = N * C
    grid = lambda meta: (nc, (spatial + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])
    _mish_tanh_bias_kernel[grid](x, bias, out, spatial, C)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # Run conv without bias, fold bias into elementwise kernel
        y = F.conv3d(x, self.conv.weight, bias=None,
                     stride=self.conv.stride, padding=self.conv.padding,
                     dilation=self.conv.dilation, groups=self.conv.groups)
        return mish_tanh_bias(y, self.conv.bias)