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
    key=['n_elements', 'C', 'spatial'],
)
@triton.jit
def _mish_tanh_kernel(
    x_ptr, bias_ptr, out_ptr, n_elements, C, spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    ch = (offsets // spatial) % C
    b = tl.load(bias_ptr + ch, mask=mask, other=0.0)
    x = x + b
    # mish: x * tanh(softplus(x))
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    tanh_sp = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    mish = x * tanh_sp
    out = 2.0 * tl.sigmoid(2.0 * mish) - 1.0
    tl.store(out_ptr + offsets, out, mask=mask)


def mish_tanh_bias(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    C = x.shape[1]
    spatial = x.shape[2] * x.shape[3] * x.shape[4]
    grid = lambda meta: ((n + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _mish_tanh_kernel[grid](x, bias, out, n, C, spatial)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self._stride = stride
        self._padding = padding

    def forward(self, x):
        y = F.conv3d(x, self.conv.weight, bias=None, stride=self._stride, padding=self._padding)
        return mish_tanh_bias(y, self.conv.bias)