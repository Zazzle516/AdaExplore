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
    CHANNELS_LAST: tl.constexpr,
    add_value: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index
    if CHANNELS_LAST:
        c = offsets % C
    else:
        c = (offsets // HW) % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x = x + b
    # Numerically stable softplus: max(x,0) + log(1+exp(-|x|))
    ax = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-ax))
    # tanh via exp
    e2 = tl.exp(2.0 * sp)
    th = (e2 - 1.0) / (e2 + 1.0)
    y = x * th
    y = y + add_value
    # Hardtanh
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    is_cl = x.is_contiguous(memory_format=torch.channels_last)
    if is_cl:
        x = x.contiguous(memory_format=torch.channels_last)
        out = torch.empty_like(x, memory_format=torch.channels_last)
    else:
        x = x.contiguous()
        out = torch.empty_like(x)
    n = x.numel()
    N, C, H, W = x.shape
    HW = H * W
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_epilogue_kernel[grid](
        x, bias, out, n, HW, C,
        CHANNELS_LAST=is_cl,
        add_value=float(add_value), scale=float(scale),
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        # Save bias separately and disable bias in the conv
        bias = self.conv_transpose.bias.detach().clone()
        self.conv_transpose.bias = None
        self.register_buffer("ct_bias", bias)
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = x.contiguous(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        x = fused_epilogue(x, self.ct_bias, self.add_value, self.scale)
        return x