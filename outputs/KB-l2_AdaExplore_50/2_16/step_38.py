import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_SIZE": 16384}, num_warps=8, num_stages=3),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_epilogue_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    add_value,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Stable softplus: log(1+exp(-|x|)) + max(x,0)
    sp = tl.log(1.0 + tl.exp(-tl.abs(x))) + tl.maximum(x, 0.0)
    e2 = tl.exp(2.0 * sp)
    tanh_sp = 1.0 - 2.0 / (e2 + 1.0)
    mish = x * tanh_sp
    y = mish + add_value
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: ((n + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    fused_epilogue_kernel[grid](x, out, n, float(add_value), float(scale))
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # Use channels_last for faster cuDNN path
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        x = fused_epilogue(x, self.add_value, self.scale)
        return x