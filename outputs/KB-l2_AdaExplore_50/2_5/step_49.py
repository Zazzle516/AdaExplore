import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _bias_tanh_kernel(
    x_ptr, b_ptr, out_ptr,
    C, HW, total,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c_idx = (offs // HW) % C
    b = tl.load(b_ptr + c_idx, mask=mask, other=0.0)
    y = x - b
    e2x = tl.exp(2.0 * y)
    out = (e2x - 1.0) / (e2x + 1.0)
    tl.store(out_ptr + offs, out, mask=mask)


def fused_bias_tanh(x, bias):
    N, C, H, W = x.shape
    HW = H * W
    total = N * C * HW
    out = torch.empty_like(x)
    BLOCK = 4096
    grid = ((total + BLOCK - 1) // BLOCK,)
    _bias_tanh_kernel[grid](x, bias, out, C, HW, total, BLOCK=BLOCK, num_warps=4, num_stages=2)
    return out


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape, stride=2, padding=1, output_padding=1):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        b = self.bias.view(-1)
        if not b.is_contiguous():
            b = b.contiguous()
        return fused_bias_tanh(x, b)