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
    inner_size,
    channels,
    add_value: tl.constexpr,
    scale: tl.constexpr,
    CHANNELS_LAST: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # Compute channel index for bias
    if CHANNELS_LAST:
        c_idx = offsets % channels
    else:
        c_idx = (offsets // inner_size) % channels
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b
    # Mish via sigmoid: tanh(sp) = 2*sigmoid(2*sp) - 1
    abs_x = tl.abs(x)
    sp = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-abs_x))
    th = 2.0 * tl.sigmoid(2.0 * sp) - 1.0
    y = x * th
    y = y + add_value
    # Hardtanh
    y = tl.minimum(tl.maximum(y, -1.0), 1.0)
    y = y * scale
    tl.store(out_ptr + offsets, y, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor, add_value: float, scale: float) -> torch.Tensor:
    N, C, H, W = x.shape
    channels_last = x.is_contiguous(memory_format=torch.channels_last)
    if channels_last:
        out = torch.empty_like(x, memory_format=torch.channels_last)
    else:
        x = x.contiguous()
        out = torch.empty_like(x)
    n = x.numel()
    inner_size = H * W
    BLOCK_SIZE = 2048
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    fused_epilogue_kernel[grid](
        x, bias, out, n, inner_size, C,
        add_value=float(add_value), scale=float(scale),
        CHANNELS_LAST=channels_last,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, add_value, scale):
        super(ModelNew, self).__init__()
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, bias=True)
        # Extract bias and disable it on the conv to fold into epilogue
        self.weight = nn.Parameter(conv.weight.data.clone().to(memory_format=torch.channels_last))
        self.bias = nn.Parameter(conv.bias.data.clone())
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.add_value = add_value
        self.scale = scale

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = F.conv_transpose2d(
            x, self.weight, bias=None,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding,
        )
        x = fused_epilogue(x, self.bias, self.add_value, self.scale)
        return x