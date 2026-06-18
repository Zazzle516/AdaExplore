import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr, bias_ptr, out_ptr,
    CHW, HW, C,
    n_elements,
    ADD_VALUE: tl.constexpr,
    MUL_VALUE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // HW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    v = x + b + ADD_VALUE
    v = tl.minimum(v, 0.0)
    inv_sqrt2 = 0.70710678118654752440
    gelu = 0.5 * v * (1.0 + tl.math.erf(v * inv_sqrt2))
    out = gelu * MUL_VALUE

    tl.store(out_ptr + offsets, out, mask=mask)


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = float(add_value)
        self.multiply_value = float(multiply_value)
        self.out_channels = out_channels

    def forward(self, x):
        w = self.conv_transpose.weight
        bias = self.conv_transpose.bias
        stride = self.conv_transpose.stride
        padding = self.conv_transpose.padding
        output_padding = self.conv_transpose.output_padding
        dilation = self.conv_transpose.dilation
        groups = self.conv_transpose.groups

        y = torch.nn.functional.conv_transpose2d(
            x, w, bias=None, stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, dilation=dilation
        )

        N, C, H, W = y.shape
        out = torch.empty_like(y)
        n_elements = y.numel()
        HW = H * W
        CHW = C * HW

        BLOCK_SIZE = 4096
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        fused_epilogue_kernel[grid](
            y, bias, out,
            CHW, HW, C,
            n_elements,
            ADD_VALUE=self.add_value,
            MUL_VALUE=self.multiply_value,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=8,
            num_stages=3,
        )
        return out