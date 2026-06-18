import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr, out_ptr,
    HW,
    const_eff_ptr,
    bias_scaled_ptr,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    nc = tl.program_id(0)
    tile = tl.program_id(1)
    base = nc * HW
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    # scalar per-program loads
    ce = tl.load(const_eff_ptr + nc % tl.num_programs(0))  # placeholder
    # The above is wrong: we need channel index, not nc. Recompute via passed C.
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    y = x  # will be overwritten below
    tl.store(out_ptr + base + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def fused_epilogue_kernel_v2(
    x_ptr, out_ptr,
    HW, C,
    const_eff_ptr,
    bias_scaled_ptr,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    nc = tl.program_id(0)
    tile = tl.program_id(1)
    c = nc % C
    base = nc * HW
    offsets = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    ce = tl.load(const_eff_ptr + c)
    bs = tl.load(bias_scaled_ptr + c)
    x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
    y = tl.minimum(x * scaling_factor + bs, ce)
    tl.store(out_ptr + base + offsets, y, mask=mask)


def fused_epilogue(x, bias, constant_value, scaling_factor):
    x = x.contiguous()
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    bias_flat = bias.contiguous().view(-1).to(x.dtype)
    const_eff = (bias_flat + constant_value) * scaling_factor
    bias_scaled = bias_flat * scaling_factor
    grid = lambda meta: (N * C, (HW + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"])
    fused_epilogue_kernel_v2[grid](
        x, out,
        HW, C,
        const_eff, bias_scaled, float(scaling_factor),
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, constant_value, bias_shape, scaling_factor):
        super(ModelNew, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.constant_value = constant_value
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor

    def forward(self, x):
        x = self.conv(x)
        x = fused_epilogue(x, self.bias, self.constant_value, self.scaling_factor)
        return x