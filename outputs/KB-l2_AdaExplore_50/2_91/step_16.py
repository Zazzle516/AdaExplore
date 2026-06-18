import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_softmax_bias_scale_sigmoid_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C, HW,
    scaling_factor: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(1)
    pid_hw = tl.program_id(0)

    offs_c = tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask_c = offs_c < C
    mask_hw = offs_hw < HW

    base = pid_n * C * HW
    # x[n, c, hw] -> base + c*HW + hw
    x_ptrs = x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :]
    mask = mask_c[:, None] & mask_hw[None, :]

    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    x_f = x.to(tl.float32)

    max_val = tl.max(x_f, axis=0)  # [BLOCK_HW]
    e = tl.exp(x_f - max_val[None, :])
    e = tl.where(mask, e, 0.0)
    sum_e = tl.sum(e, axis=0)  # [BLOCK_HW]
    sm = e / sum_e[None, :]

    b = tl.load(bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    y = (sm + b[:, None]) * scaling_factor
    out = 1.0 / (1.0 + tl.exp(-y))

    tl.store(x_ptr + base + offs_c[:, None] * HW + offs_hw[None, :] - x_ptr + out_ptr,
             out.to(x.dtype), mask=mask)


def fused_softmax_bias_scale_sigmoid(x, bias, scaling_factor):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)

    BLOCK_C = triton.next_power_of_2(C)
    BLOCK_HW = 128
    grid = (triton.cdiv(HW, BLOCK_HW), N)

    fused_softmax_bias_scale_sigmoid_kernel[grid](
        x, bias, out,
        N, C, HW,
        float(scaling_factor),
        BLOCK_C=BLOCK_C,
        BLOCK_HW=BLOCK_HW,
        num_warps=8,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                                                  stride=stride, padding=padding,
                                                  output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.scaling_factor = scaling_factor
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding

    def forward(self, x):
        y = F.conv_transpose2d(
            x, self.conv_transpose.weight, self.conv_transpose.bias,
            stride=self.stride, padding=self.padding,
            output_padding=self.output_padding,
        )
        y = y.contiguous()
        bias_flat = self.bias.view(-1).contiguous()
        return fused_softmax_bias_scale_sigmoid(y, bias_flat, self.scaling_factor)