import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import math


@triton.jit
def fused_epilogue_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    HW,
    C,
    ADD_VALUE: tl.constexpr,
    SCALE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index = (offsets // HW) % C
    c_idx = (offsets // HW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b
    # Mish: x * tanh(softplus(x)) ; cheaper softplus
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(x)))
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = x * th
    y = y + ADD_VALUE
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * SCALE
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty_like(x)
    n = x.numel()
    N, C, H, W = x.shape
    HW = H * W
    BLOCK_SIZE = 4096
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_epilogue_kernel[grid](
        x, bias, out, n, HW, C,
        ADD_VALUE=float(add_value), SCALE=float(scale),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=8, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.add_value = add_value
        self.scale = scale
        # Save bias separately and disable bias in conv for fused add
        self._bias = nn.Parameter(self.conv_transpose.bias.data.clone())
        self.conv_transpose.bias = None

    def forward(self, x):
        x = self.conv_transpose(x)
        x = fused_epilogue(x, self._bias, self.add_value, self.scale)
        return x