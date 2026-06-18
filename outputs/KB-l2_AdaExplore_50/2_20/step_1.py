import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def fused_epilogue_kernel(
    x_ptr,      # conv output, shape [N, C, D, H, W]
    bias_ptr,   # bias, shape [C]
    out_ptr,    # output
    n_elements,
    C, DHW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # compute channel index: (offsets // DHW) % C
    c_idx = (offsets // DHW) % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # original_x = x
    # step1 = x + b
    # step2 = step1 + x = 2x + b
    # step3 = step2 * x = (2x + b) * x
    # step4 = step3 + x
    out = (2.0 * x + b) * x + x
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_epilogue(x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    bias_flat = bias.contiguous().view(-1)
    out = torch.empty_like(x)
    n_elements = x.numel()
    N, C, D, H, W = x.shape
    DHW = D * H * W
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    fused_epilogue_kernel[grid](
        x, bias_flat, out, n_elements, C, DHW, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4, num_stages=2,
    )
    return out


class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, output_padding=output_padding,
        )
        self.bias = nn.Parameter(torch.randn(bias_shape))

    def forward(self, x):
        x = self.conv_transpose(x)
        return fused_epilogue(x, self.bias)