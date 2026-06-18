import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


from triton.language.extra import libdevice as _libdev


@triton.jit
def _bias_tanh_2d_kernel(
    x_ptr, b_ptr, out_ptr,
    C, HW,
    BLOCK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_c = pid_nc % C
    base = pid_nc * HW
    offs = pid_h * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + pid_c)
    y = x - b
    out = _libdev.tanh(y)
    tl.store(out_ptr + base + offs, out, mask=mask)


def fused_bias_tanh_nchw(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    BLOCK = 8192
    grid = (N * C, (HW + BLOCK - 1) // BLOCK)
    _bias_tanh_2d_kernel[grid](x, bias, out, C, HW, BLOCK=BLOCK, num_warps=8, num_stages=2)
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.kernel_size = kernel_size

    def forward(self, x):
        # Use cuDNN ConvTranspose2d (highly tuned), then fused bias-subtract + tanh.
        x = self.conv_transpose(x)
        x = x.contiguous()
        b = self.bias.view(-1).contiguous()
        return fused_bias_tanh_nchw(x, b)