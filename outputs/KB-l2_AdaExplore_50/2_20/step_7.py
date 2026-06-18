import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr,      # conv output, shape [N, C, D, H, W] contiguous
    bias_ptr,   # bias, shape [C]
    out_ptr,    # output
    DHW,
    BLOCK_SIZE: tl.constexpr,
):
    nc_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    # Load bias once per program (scalar)
    c_idx = nc_id % tl.num_programs(0)  # not needed; use direct
    # We need C to compute channel; pass it via second pid grouping.
    # Instead: nc_id encodes n*C + c, so c = nc_id % C. We pass C as constexpr-ish via grid.
    # Simpler: bias index = nc_id % C, but C unknown here. So load via passing channel directly.
    # Use a separate channel_ptr: bias_ptr is indexed by (nc_id % C); we pass C as arg.
    pass


@triton.jit
def fused_epilogue_kernel_v2(
    x_ptr, bias_ptr, out_ptr,
    C, DHW,
    BLOCK_SIZE: tl.constexpr,
):
    nc_id = tl.program_id(0)
    tile_id = tl.program_id(1)
    c = nc_id % C
    b = tl.load(bias_ptr + c)

    base = nc_id * DHW
    offs = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < DHW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    out = (2.0 * x + b) * x + x
    tl.store(out_ptr + base + offs, out, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    N, C, D, H, W = x.shape
    DHW = D * H * W
    BLOCK_SIZE = 2048
    num_tiles = (DHW + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (N * C, num_tiles)
    fused_epilogue_kernel_v2[grid](
        x, bias_flat, out, C, DHW, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8, num_stages=3,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.bias)