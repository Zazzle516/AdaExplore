import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK': 2048}, num_warps=4, num_stages=2),
    ],
    key=['SPATIAL'],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, scale_ptr, bias_ptr, out_ptr,
    NC, SPATIAL,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_s = pid_s * BLOCK + tl.arange(0, BLOCK)
    mask = offs_s < SPATIAL

    base = pid_nc * SPATIAL
    s = tl.load(scale_ptr + pid_nc, cache_modifier=".ca")
    b = tl.load(bias_ptr + pid_nc, cache_modifier=".ca")

    x = tl.load(x_ptr + base + offs_s, mask=mask, other=0.0)
    v = x * s
    # fast tanh via sigmoid: tanh(v) = 2*sigmoid(2v) - 1
    t = 2.0 * tl.sigmoid(2.0 * v) - 1.0
    y = t * b
    out = tl.sigmoid(y)

    tl.store(out_ptr + base + offs_s, out, mask=mask)


def fused_epilogue(x, scale, bias):
    x = x.contiguous()
    N, C, D, H, W = x.shape
    out = torch.empty_like(x)
    SPATIAL = D * H * W
    NC = N * C
    # broadcast scale/bias to length C, then tile to NC by repeating
    scale_c = scale.contiguous().view(-1)  # length C
    bias_c = bias.contiguous().view(-1)
    # expand to NC by repeating per batch
    scale_nc = scale_c.repeat(N)
    bias_nc = bias_c.repeat(N)
    grid = lambda meta: (NC, triton.cdiv(SPATIAL, meta['BLOCK']))
    fused_epilogue_kernel[grid](
        x, scale_nc, bias_nc, out,
        NC, SPATIAL,
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