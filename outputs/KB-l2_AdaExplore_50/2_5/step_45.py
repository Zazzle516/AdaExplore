import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


torch.backends.cudnn.benchmark = True


@triton.jit
def _bias_tanh_channels_last_kernel(
    x_ptr, b_ptr, out_ptr,
    TOTAL, C,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < TOTAL
    c_idx = offs % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    e2x = tl.exp(2.0 * y)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.jit
def _bias_tanh_nchw_kernel(
    x_ptr, b_ptr, out_ptr,
    N, C, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < HW

    base = pid_n * C * HW + pid_c * HW
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + pid_c)
    y = x - b
    e2x = tl.exp(2.0 * y)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + base + offs, out, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    TOTAL = N * C * HW
    # Detect channels_last layout: stride(C-dim) == 1
    if x.stride(1) == 1 and x.stride(3) == C:
        out = torch.empty_like(x)
        BLOCK = 8192
        grid = ((TOTAL + BLOCK - 1) // BLOCK,)
        _bias_tanh_channels_last_kernel[grid](
            x, bias, out, TOTAL, C, BLOCK=BLOCK, num_warps=4, num_stages=2
        )
        return out
    else:
        x = x.contiguous()
        out = torch.empty_like(x)
        BLOCK = 4096 if HW >= 4096 else 1024
        grid = ((HW + BLOCK - 1) // BLOCK, C, N)
        _bias_tanh_nchw_kernel[grid](
            x, bias, out, N, C, HW, BLOCK=BLOCK, num_warps=8, num_stages=3
        )
        return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        torch.backends.cudnn.benchmark = True
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        # Fold conv's bias into our bias term (still runs conv; just avoids redundant bias add)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        with torch.no_grad():
            conv_b = self.conv_transpose.bias.detach().view(-1, 1, 1)
            # original: y = (conv_no_bias + conv_b) - self.bias  ==>  conv_no_bias - (self.bias - conv_b)
            self.bias.data = self.bias.data - conv_b
        self.conv_transpose.bias = None
        # Convert weight to channels_last for faster cuDNN ConvTranspose2d
        self.conv_transpose = self.conv_transpose.to(memory_format=torch.channels_last)

    def forward(self, x):
        x = x.to(memory_format=torch.channels_last)
        x = self.conv_transpose(x)
        b = self.bias.reshape(-1).contiguous()
        return fused_bias_tanh(x, b)